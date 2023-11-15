# import the necessary packages
from tensorflow import keras

from TheLazyCoder import social_distancing_config as config
from TheLazyCoder.detection import detect_people
from scipy.spatial import distance as dist
import argparse
from keras.applications.mobilenet_v2 import preprocess_input
from keras.utils import img_to_array
from keras.models import load_model
import datetime
import json
import pdfkit
import winsound
import subprocess
import webview

import pyttsx3 as voice
from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2
import os
import sqlite3

from flask import Flask, flash, render_template, Response, request, redirect, url_for, make_response, session, g
from database import get_database
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
window = webview.create_window('MonitorLab', app)
app.config['SECRET_KEY'] = os.urandom(24)  # for storing a session

violation = 0
social_violation = 0
total_violation = 0
compliance = 0
start_date = None
end_date = None

Class_Name = None
Report_Type = None
Report_Date = None
Recommendation = None


def get_current_user():
    user = None

    if 'user' in session:
        user = session['user']

        db = get_database()
        user_cur = db.execute('select * from users where Username = ?', [user])
        user = user_cur.fetchone()

    return user


def get_current_room():
    rooms = None

    if 'room' in session:
        rooms = session['rooms']

        db = get_database()
        user_cur = db.execute('select * from classroom where class_name = ?', [rooms])
        rooms = user_cur.fetchone()

    return rooms


def get_current_admin():
    admins = None

    if 'admins' in session:
        admins = session['admins']

        db = get_database()
        admin_cur = db.execute('select * from admin where AdminId = ?', [admins])
        admins = admin_cur.fetchone()

    return admins


def detect_and_predict_mask(frame, faceNet, maskNet):
    global violation
    global compliance
    # grab the dimensions of the frame and then construct a blob
    # from it
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
                                 (104.0, 177.0, 123.0))

    # pass the blob through the network and obtain the face detections
    faceNet.setInput(blob)
    detections = faceNet.forward()
    print(detections.shape)

    # initialize our list of faces, their corresponding locations,
    # and the list of predictions from our face mask network
    faces = []
    locs = []
    preds = []

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with
        # the detection
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the confidence is
        # greater than the minimum confidence
        if confidence > 0.5:
            # compute the (x, y)-coordinates of the bounding box for
            # the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # ensure the bounding boxes fall within the dimensions of
            # the frame
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            # extract the face ROI, convert it from BGR to RGB channel
            # ordering, resize it to 224x224, and preprocess it
            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)

            # add the face and bounding boxes to their respective
            # lists
            faces.append(face)
            locs.append((startX, startY, endX, endY))

    # only make a predictions if at least one face was detected
    if len(faces) > 0:
        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces, batch_size=32)

    for pred in preds:
        (mask, withoutMask) = pred

    for i in faces:
        if mask < withoutMask:
            #frequency = 1000  # Set frequency to 1000 Hz
            #duration = 700  # Set duration to 500 ms

            #winsound.Beep(frequency, duration)

            violation = len(preds)

        elif mask > withoutMask:
            compliance = len(preds)

    # return a 2-tuple of the face locations and their corresponding
    # locations
    return (locs, preds)


# load our serialized face detector model from disk
prototxtPath = r"face_detector\deploy.prototxt"
weightsPath = r"face_detector\res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# load the face mask detector model from disk
maskNet = load_model("mask-detector-model.model")

# initialize the video stream
print("[INFO] starting video stream...")
vs = cv2.VideoCapture(1)



def generate_frames():
    global violation
    global compliance

    while True:
        success, frame = vs.read()

        # detect faces in the frame and determine if they are wearing a
        # face mask or not
        (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)
        # loop over the detected face locations and their corresponding
        # locations
        for (box, pred) in zip(locs, preds):
            # unpack the bounding box and predictions
            (startX, startY, endX, endY) = box
            (mask, withoutMask) = pred

            # determine the class label and color we'll use to draw
            # the bounding box and text
            label = "Mask" if mask > withoutMask else "No Mask"
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

            # include the probability in the label
            label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

            # adding voice assistant
            # voice.speak("cleared") if mask > withoutMask else voice.speak("No mask please put on a mask")

            # display the label and bounding box rectangle on the output
            # frame
            cv2.putText(frame, label, (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

            # adding voice assistant
            voice.speak("cleared") if mask > withoutMask else voice.speak("No mask please put on a mask")

        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame +
               b'\r\n\r\n')


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", type=str, default="",
                help="path to (optional) input video file")
ap.add_argument("-o", "--output", type=str, default="",
                help="path to (optional) output video file")
ap.add_argument("-d", "--display", type=int, default=1,
                help="whether or not output frame should be displayed")
args = vars(ap.parse_args())

# load the COCO class labels our YOLO model was trained on
labelsPath = os.path.sep.join([config.MODEL_PATH, "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

# derive the paths to the YOLO weights and model configuration
weightsPath = r"yolov3.weights"
configPath = r"yolov3.cfg"

# load our YOLO object detector trained on COCO dataset (80 classes)
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

# check if we are going to use GPU
if config.USE_GPU:
    # set CUDA as the preferable backend and target
    print("[INFO] setting preferable backend and target to CUDA...")
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# determine only the *output* layer names that we need from YOLO
ln = net.getLayerNames()
ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

# initialize the video stream and pointer to output video file
print("[INFO] accessing video stream...")
sd = cv2.VideoCapture(2)
writer = None


def generate_social():
    global social_violation
    while True:
        # read the next frame from the file
        (grabbed, frame) = sd.read()

        # if the frame was not grabbed, then we have reached the end
        # of the stream

        # resize the frame and then detect people (and only people) in it
        frame = imutils.resize(frame, width=700)
        results = detect_people(frame, net, ln,
                                personIdx=LABELS.index("person"))

        # initialize the set of indexes that violate the minimum social
        # distance
        violate = set()

        # ensure there are *at least* two people detections (required in
        # order to compute our pairwise distance maps)
        if len(results) >= 2:
            # extract all centroids from the results and compute the
            # Euclidean distances between all pairs of the centroids
            centroids = np.array([r[2] for r in results])
            D = dist.cdist(centroids, centroids, metric="euclidean")

            # loop over the upper triangular of the distance matrix
            for i in range(0, D.shape[0]):
                for j in range(i + 1, D.shape[1]):
                    # check to see if the distance between any two
                    # centroid pairs is less than the configured number
                    # of pixels
                    if D[i, j] < config.MIN_DISTANCE:
                        # update our violation set with the indexes of
                        # the centroid pairs
                        violate.add(i)
                        violate.add(j)

        # loop over the results
        for (i, (prob, bbox, centroid)) in enumerate(results):
            # extract the bounding box and centroid coordinates, then
            # initialize the color of the annotation
            (startX, startY, endX, endY) = bbox
            (cX, cY) = centroid
            color = (0, 255, 0)

            # if the index pair exists within the violation set, then
            # update the color
            if i in violate:
                color = (0, 0, 255)

            # draw (1) a bounding box around the person and (2) the
            # centroid coordinates of the person,
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
            cv2.circle(frame, (cX, cY), 5, color, 1)

        # draw the total number of social distancing violations on the
        # output frame
        text = "Social Distancing Violations: {}".format(len(violate))
        social_violation = len(violate)
        cv2.putText(frame, text, (10, frame.shape[0] - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 3)

        if not grabbed:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame +
               b'\r\n\r\n')


@app.route('/')
def land():
    return render_template('land.html')


@app.route('/login', methods=["POST", "GET"])
def login():
    error = None

    if request.method == "POST":
        uname = request.form['uname']
        password = request.form['password']

        db = get_database()
        user_cursor = db.execute('select * from users where Username = ?', [uname])
        user = user_cursor.fetchone()

        if user:
            if check_password_hash(user['password'], password):

                session['user'] = user['Username']
                flash("Successfully logged in", "success")
                return redirect(url_for('room'))
            else:
                flash("Password did not match or User does not exist", "error")

        else:
            flash("Password did not match or User does not exist", "error")

    return render_template('login.html')


@app.route('/register', methods=["POST", "GET"])
def register():
    if request.method == "POST":
        db = get_database()
        fname = request.form['fname']
        lname = request.form['lname']
        email = request.form['email']
        phone = request.form['phone']
        uname = request.form['uname']
        nid = request.form['nid']
        password = request.form['password']
        hash_password = generate_password_hash(password)

        dbuser = db.execute('select * from users where Username = ?', [uname])
        existing_user = dbuser.fetchone()

        if existing_user:
            return render_template('register.html', registererror='Username already taken, choose another username')

        db.execute('insert into users( Username, password) values (?,?)', [uname, hash_password])
        db.execute('insert into employee( fname, lname,Username,email,phone,nationalID) values (?,?,?,?,?,?)',
                   [fname, lname, uname, email, phone, nid])
        db.commit()
        return redirect(url_for('home'))
    return render_template('register.html')


@app.route('/home')
def home():
    global start_date
    user = get_current_user()
    rooms = get_current_room()
    start_date = datetime.datetime.now()

    return render_template('home.html', user=user, rooms=rooms)


@app.route('/room')
def room():
    user = get_current_user()
    db = get_database()
    class_cursor = db.execute('select * from classroom limit 4')
    class_data = class_cursor.fetchall()
    user_cursor = db.execute('select * from classroom')
    data = user_cursor.fetchall()

    pie_total_data = db.execute('select sum(v.total_violations), c.class_name from violations v,'
                                'classroom c where v.class_name = c.class_name group by v.class_name')

    pie_total_label = db.execute('select c.class_name, sum(v.violations) from violations v,'
                                 'classroom c where v.class_name = c.class_name group by v.class_name')

    pie_data = pie_total_data.fetchall()
    pie_violation_data = []
    for total_amount, _ in pie_data:
        pie_violation_data.append(total_amount)

    pie_label = pie_total_label.fetchall()
    pie_violation_label = []
    for total_amount, _ in pie_label:
        pie_violation_label.append(total_amount)

    # monthly bar data
    bar_total_data = db.execute('select sum(v.violations), strftime("%m", start_of_detection) from violations v,'
                                ' classroom c where v.class_name = c.class_name group by strftime("%m", start_of_detection)')

    bar_total_data2 = db.execute(
        'select sum(v.social_violations), strftime("%m", start_of_detection) from violations v,'
        ' classroom c where v.class_name = c.class_name group by strftime("%m", start_of_detection)')

    bar_total_label = db.execute(
        'select CASE strftime("%m", start_of_detection) WHEN "01" THEN "January" WHEN "02" THEN "February" WHEN "03" THEN "March" WHEN "04" THEN "+'
        'April" WHEN "05" THEN "May"'
        'WHEN "06" THEN "June" WHEN "07" THEN "July" WHEN "08" THEN "August" WHEN "09" THEN "September" WHEN "10" THEN "October" WHEN "11" THEN "November"'
        'WHEN "12" THEN "December" END as month_name, sum(v.violations) from violations v, classroom c where v.class_name = c.class_name group by strftime("%m", start_of_detection)')

    bar_data = bar_total_data.fetchall()
    bar_violation_data = []
    for total_amount, _ in bar_data:
        bar_violation_data.append(total_amount)

    bar_data2 = bar_total_data2.fetchall()
    bar_violation_data2 = []
    for total_amount2, _ in bar_data2:
        bar_violation_data2.append(total_amount2)

    bar_label = bar_total_label.fetchall()
    bar_violation_label = []
    for total_amount, _ in bar_label:
        bar_violation_label.append(total_amount)

    # yearly bar data
    yr_bar_total_data = db.execute('select sum(v.violations), strftime("%y", start_of_detection) from violations v,'
                                   ' classroom c where v.class_name = c.class_name group by strftime("%y", start_of_detection)')

    yr_bar_total_data2 = db.execute(
        'select sum(v.social_violations), strftime("%y", start_of_detection) from violations v,'
        ' classroom c where v.class_name = c.class_name group by strftime("%y", start_of_detection)')

    yr_bar_total_label = db.execute('select strftime("%y", start_of_detection)  , sum(v.violations)'
                                    ' from violations v, classroom c where v.class_name = c.class_name '
                                    'group by strftime("%y", start_of_detection)')

    yr_bar_data = yr_bar_total_data.fetchall()
    yr_bar_violation_data = []
    for yr_total_amount, _ in yr_bar_data:
        yr_bar_violation_data.append(yr_total_amount)

    yr_bar_data2 = yr_bar_total_data2.fetchall()
    yr_bar_violation_data2 = []
    for yr_total_amount2, _ in yr_bar_data2:
        yr_bar_violation_data2.append(yr_total_amount2)

    yr_bar_label = yr_bar_total_label.fetchall()
    yr_bar_violation_label = []
    for yr_total_amount, _ in yr_bar_label:
        yr_bar_violation_label.append(yr_total_amount)

    return render_template('room.html', user=user, value=data, class_data=class_data,
                           pie_violation_label=json.dumps(pie_violation_label),
                           pie_violation_data=json.dumps(pie_violation_data),
                           bar_violation_label=json.dumps(bar_violation_label),
                           bar_violation_data2=json.dumps(bar_violation_data2),
                           bar_violation_data=json.dumps(bar_violation_data),
                           yr_bar_violation_data=json.dumps(yr_bar_violation_data),
                           yr_bar_violation_data2=json.dumps(yr_bar_violation_data2),
                           yr_bar_violation_label=json.dumps(yr_bar_violation_label))


@app.route('/room2', methods=["POST", "GET"])
def room2():
    global start_date
    if request.method == "POST":
        session['rooms'] = request.form['class_name']
        user = get_current_user()
        start_date = datetime.datetime.now()
        return render_template('home.html', user=user, rooms=session['rooms'])
    return redirect(url_for('room'))


@app.route('/Dashboard')
def Dashboard():
    get_current_user()
    admins = get_current_admin()
    db = get_database()
    user_cursor = db.execute('select * from employee')
    data = user_cursor.fetchall()
    return render_template('Dashboard.html', admins=admins, value=data)


@app.route('/profile/<uname>')
def profile(uname):
    user = get_current_user()
    db = get_database()
    user_cursor = db.execute('select * from employee where Username = ?', [uname])
    data = user_cursor.fetchall()
    return render_template('profile.html', user=user, value=data)

@app.route('/edit_profile/<uname>')
def edit_profile(uname):
    db = get_database()
    user = get_current_user()
    session.pop('user', None)
    user_cursor = db.execute('select * from employee where Username = ?', [uname])
    data = user_cursor.fetchall()
    return render_template('edit_profile.html', user=user, value=data)


@app.route('/admin_login', methods=["POST", "GET"])
def admin_login():
    error = None
    user = get_current_user()

    if request.method == "POST":
        admin_id = request.form['AdminId']
        password = request.form['password']

        db = get_database()
        admin_cursor = db.execute('select * from admin where AdminId = ?', [admin_id])
        admins = admin_cursor.fetchone()

        if admins:
            if admins['password'] != password:
                error = "Password did not match or AdminId does not exist"
            else:
                session['admins'] = admins['AdminId']
                return redirect(url_for('admin'))
        else:
            error = "Password did not match or AdminId does not exist"
    return render_template('admin_login.html', loginerror=error, user=user)


@app.route('/admin')
def admin():
    admins = get_current_admin()
    get_current_user()
    db = get_database()
    class_cursor = db.execute('select * from classroom limit 4')
    class_data = class_cursor.fetchall()


    pie_total_data = db.execute('select sum(v.total_violations), c.class_name from violations v,'
                                'classroom c where v.class_name = c.class_name group by v.class_name')

    pie_total_label = db.execute('select c.class_name, sum(v.violations) from violations v,'
                                 'classroom c where v.class_name = c.class_name group by v.class_name')

    pie_data = pie_total_data.fetchall()
    pie_violation_data = []
    for total_amount, _ in pie_data:
        pie_violation_data.append(total_amount)

    pie_label = pie_total_label.fetchall()
    pie_violation_label = []
    for total_amount, _ in pie_label:
        pie_violation_label.append(total_amount)

    # monthly bar data
    bar_total_data = db.execute('select sum(v.violations), strftime("%m", start_of_detection) from violations v,'
                                ' classroom c where v.class_name = c.class_name group by strftime("%m", start_of_detection)')

    bar_total_data2 = db.execute(
        'select sum(v.social_violations), strftime("%m", start_of_detection) from violations v,'
        ' classroom c where v.class_name = c.class_name group by strftime("%m", start_of_detection)')

    bar_total_label = db.execute(
        'select CASE strftime("%m", start_of_detection) WHEN "01" THEN "January" WHEN "02" THEN "February" WHEN "03" THEN "March" WHEN "04" THEN "+'
        'April" WHEN "05" THEN "May"'
        'WHEN "06" THEN "June" WHEN "07" THEN "July" WHEN "08" THEN "August" WHEN "09" THEN "September" WHEN "10" THEN "October" WHEN "11" THEN "November"'
        'WHEN "12" THEN "December" END as month_name, sum(v.violations) from violations v, classroom c where v.class_name = c.class_name group by strftime("%m", start_of_detection)')

    bar_data = bar_total_data.fetchall()
    bar_violation_data = []
    for total_amount, _ in bar_data:
        bar_violation_data.append(total_amount)

    bar_data2 = bar_total_data2.fetchall()
    bar_violation_data2 = []
    for total_amount2, _ in bar_data2:
        bar_violation_data2.append(total_amount2)

    bar_label = bar_total_label.fetchall()
    bar_violation_label = []
    for total_amount, _ in bar_label:
        bar_violation_label.append(total_amount)

    # yearly bar data
    yr_bar_total_data = db.execute('select sum(v.violations), strftime("%y", start_of_detection) from violations v,'
                                   ' classroom c where v.class_name = c.class_name group by strftime("%y", start_of_detection)')

    yr_bar_total_data2 = db.execute(
        'select sum(v.social_violations), strftime("%y", start_of_detection) from violations v,'
        ' classroom c where v.class_name = c.class_name group by strftime("%y", start_of_detection)')

    yr_bar_total_label = db.execute('select strftime("%y", start_of_detection)  , sum(v.violations)'
                                    ' from violations v, classroom c where v.class_name = c.class_name '
                                    'group by strftime("%y", start_of_detection)')

    yr_bar_data = yr_bar_total_data.fetchall()
    yr_bar_violation_data = []
    for yr_total_amount, _ in yr_bar_data:
        yr_bar_violation_data.append(yr_total_amount)

    yr_bar_data2 = yr_bar_total_data2.fetchall()
    yr_bar_violation_data2 = []
    for yr_total_amount2, _ in yr_bar_data2:
        yr_bar_violation_data2.append(yr_total_amount2)

    yr_bar_label = yr_bar_total_label.fetchall()
    yr_bar_violation_label = []
    for yr_total_amount, _ in yr_bar_label:
        yr_bar_violation_label.append(yr_total_amount)

    return render_template('admin.html', admins=admins, class_data=class_data,
                           pie_violation_label=json.dumps(pie_violation_label),
                           pie_violation_data=json.dumps(pie_violation_data),
                           bar_violation_label=json.dumps(bar_violation_label),
                           bar_violation_data2=json.dumps(bar_violation_data2),
                           bar_violation_data=json.dumps(bar_violation_data),
                           yr_bar_violation_data=json.dumps(yr_bar_violation_data),
                           yr_bar_violation_data2=json.dumps(yr_bar_violation_data2),
                           yr_bar_violation_label=json.dumps(yr_bar_violation_label))


@app.route('/adduser', methods=["POST", "GET"])
def adduser():
    admins = get_current_admin()
    if request.method == "POST":
        db = get_database()
        fname = request.form['fname']
        lname = request.form['lname']
        email = request.form['email']
        phone = request.form['phone']
        uname = request.form['uname']
        nid = request.form['nid']
        password = request.form['password']
        hash_password = generate_password_hash(password)

        dbuser = db.execute('select * from users where Username = ?', [uname])
        existing_user = dbuser.fetchone()

        if existing_user:
            return render_template('adduser.html', registererror='Username already taken, choose another username')

        db.execute('insert into users( Username, password) values (?,?)', [uname, hash_password])
        db.execute('insert into employee( fname, lname,Username,email,phone,nationalID) values (?,?,?,?,?,?)',
                   [fname, lname, uname, email, phone, nid])
        db.commit()
        return redirect(url_for('Dashboard'))
    return render_template('adduser.html', admins=admins)


@app.route('/updating', methods=["POST", "GET"])
def updating():

    if request.method == "POST":
        db = get_database()
        fname = request.form['fname']
        lname = request.form['lname']
        email = request.form['email']
        phone = request.form['phone']
        uname = request.form['uname']
        nid = request.form['nid']

        db.execute('UPDATE employee SET fname = ?, lname = ?, email = ? , phone = ? ,nationalID = ? '
                   'where  Username = ?', (fname, lname, email, phone, nid, uname))
        db.commit()

        user_cursor = db.execute('select * from employee where Username = ?', [uname])
        data = user_cursor.fetchall()

        session['user'] = uname
        return render_template('profile.html', user=session['user'], value=data)
    return redirect(url_for('edit_profile'))


@app.route('/updating2', methods=["POST", "GET"])
def updating2():
    get_current_user()

    if request.method == "POST":
        db = get_database()
        fname = request.form['fname']
        lname = request.form['lname']
        email = request.form['email']
        phone = request.form['phone']
        uname = request.form['uname']
        nid = request.form['nid']

        db.execute('UPDATE employee SET fname = ?, lname = ?, email = ? , phone = ? ,nationalID = ? '
                   'where  Username = ?', (fname, lname, email, phone, nid, uname))
        db.commit()

        admins = get_current_admin()
        user_cursor = db.execute('select * from employee')
        data = user_cursor.fetchall()
        return render_template('Dashboard.html', admins=admins, value=data)
    return redirect(url_for('Update'))



@app.route('/change_password')
def change_password():
    user = get_current_user()
    return render_template('change_password.html', user=user)


@app.route('/graph')
def graph():
    admins = get_current_admin()
    return render_template('graph.html', admins=admins)


@app.route('/change_admin_password')
def change_admin_password():
    admins = get_current_admin()
    return render_template('change_admin_password.html', admins=admins)


@app.route('/classroom', methods=["POST", "GET"])
def classroom():
    admins = get_current_admin()
    if request.method == "POST":
        db = get_database()
        class_name = request.form['class_name']
        total_cameras = request.form['total_cameras']
        total_v = request.form['total_v']

        db.execute('insert into classroom( class_name, total_cameras, total_violations) values (?,?,?)',
                   [class_name, total_cameras, total_v])
        db.commit()
        return redirect(url_for('classes'))
    return render_template('classroom.html', admins=admins)


@app.route('/classes')
def classes():
    admins = get_current_admin()
    db = get_database()
    user_cursor = db.execute(' select c.class_name,sum(v.total_violations) from violations v, '
                             'classroom c where v.class_name = c.class_name group by c.class_name'
                             ' order by sum(v.total_violations)')
    data = user_cursor.fetchall()
    return render_template('classes.html', admins=admins, value=data)


@app.route('/logout')
def logout():
    global end_date
    global start_date
    global violation
    global social_violation
    global total_violation

    total_violation = social_violation + violation
    user = get_current_user()
    end_date = datetime.datetime.now()
    db = get_database()
    user_cursor = db.execute('select * from classroom')
    data = user_cursor.fetchall()
    class_cursor = db.execute('select * from classroom limit 4')
    class_data = class_cursor.fetchall()
    db.execute(
        'insert into violations ( class_name, start_of_detection, end_of_detection, violations,social_violations, '
        'total_violations) values (?,?,?,?,?,?)',
        [session['rooms'], start_date, end_date, violation, social_violation, total_violation])

    db.commit()
    pie_total_data = db.execute('select sum(v.total_violations), c.class_name from violations v,'
                                'classroom c where v.class_name = c.class_name group by v.class_name')

    pie_total_label = db.execute('select c.class_name, sum(v.violations) from violations v,'
                                 'classroom c where v.class_name = c.class_name group by v.class_name')

    pie_data = pie_total_data.fetchall()
    pie_violation_data = []
    for total_amount, _ in pie_data:
        pie_violation_data.append(total_amount)

    pie_label = pie_total_label.fetchall()
    pie_violation_label = []
    for total_amount, _ in pie_label:
        pie_violation_label.append(total_amount)

    # monthly bar data
    bar_total_data = db.execute('select sum(v.violations), strftime("%m", start_of_detection) from violations v,'
                                ' classroom c where v.class_name = c.class_name group by strftime("%m", start_of_detection)')

    bar_total_data2 = db.execute(
        'select sum(v.social_violations), strftime("%m", start_of_detection) from violations v,'
        ' classroom c where v.class_name = c.class_name group by strftime("%m", start_of_detection)')

    bar_total_label = db.execute(
        'select CASE strftime("%m", start_of_detection) WHEN "01" THEN "January" WHEN "02" THEN "February" WHEN "03" THEN "March" WHEN "04" THEN "+'
        'April" WHEN "05" THEN "May"'
        'WHEN "06" THEN "June" WHEN "07" THEN "July" WHEN "08" THEN "August" WHEN "09" THEN "September" WHEN "10" THEN "October" WHEN "11" THEN "November"'
        'WHEN "12" THEN "December" END as month_name, sum(v.violations) from violations v, classroom c where v.class_name = c.class_name group by strftime("%m", start_of_detection)')

    bar_data = bar_total_data.fetchall()
    bar_violation_data = []
    for total_amount, _ in bar_data:
        bar_violation_data.append(total_amount)

    bar_data2 = bar_total_data2.fetchall()
    bar_violation_data2 = []
    for total_amount2, _ in bar_data2:
        bar_violation_data2.append(total_amount2)

    bar_label = bar_total_label.fetchall()
    bar_violation_label = []
    for total_amount, _ in bar_label:
        bar_violation_label.append(total_amount)

    # yearly bar data
    yr_bar_total_data = db.execute('select sum(v.violations), strftime("%y", start_of_detection) from violations v,'
                                   ' classroom c where v.class_name = c.class_name group by strftime("%y", start_of_detection)')

    yr_bar_total_data2 = db.execute(
        'select sum(v.social_violations), strftime("%y", start_of_detection) from violations v,'
        ' classroom c where v.class_name = c.class_name group by strftime("%y", start_of_detection)')

    yr_bar_total_label = db.execute('select strftime("%y", start_of_detection)  , sum(v.violations)'
                                    ' from violations v, classroom c where v.class_name = c.class_name '
                                    'group by strftime("%y", start_of_detection)')

    yr_bar_data = yr_bar_total_data.fetchall()
    yr_bar_violation_data = []
    for yr_total_amount, _ in yr_bar_data:
        yr_bar_violation_data.append(yr_total_amount)

    yr_bar_data2 = yr_bar_total_data2.fetchall()
    yr_bar_violation_data2 = []
    for yr_total_amount2, _ in yr_bar_data2:
        yr_bar_violation_data2.append(yr_total_amount2)

    yr_bar_label = yr_bar_total_label.fetchall()
    yr_bar_violation_label = []
    for yr_total_amount, _ in yr_bar_label:
        yr_bar_violation_label.append(yr_total_amount)

    return render_template('room.html', user=user, value=data, class_data=class_data,
                           pie_violation_label=json.dumps(pie_violation_label),
                           pie_violation_data=json.dumps(pie_violation_data),
                           bar_violation_label=json.dumps(bar_violation_label),
                           bar_violation_data2=json.dumps(bar_violation_data2),
                           bar_violation_data=json.dumps(bar_violation_data),
                           yr_bar_violation_data=json.dumps(yr_bar_violation_data),
                           yr_bar_violation_data2=json.dumps(yr_bar_violation_data2),
                           yr_bar_violation_label=json.dumps(yr_bar_violation_label))


@app.route('/delete/<uname>')
def delete(uname):
    db = get_database()
    admins = get_current_admin()
    session.pop('user', None)
    db.execute('delete from users where Username = ?', [uname])
    db.execute('delete from employee where Username = ?', [uname])
    db.commit()
    user_cursor = db.execute('select * from employee')
    data = user_cursor.fetchall()
    return render_template('Dashboard.html', admins=admins, value=data)


@app.route('/Update/<uname>')
def Update(uname):
    get_current_user()
    db = get_database()
    admins = get_current_admin()
    session.pop('user', None)
    user_cursor2 = db.execute('select * from users where Username = ?', [uname])
    data2 = user_cursor2.fetchall()
    user_cursor = db.execute('select * from employee where Username = ?', [uname])
    data = user_cursor.fetchall()
    return render_template('Update.html', admins=admins, value=data, value2=data2)

@app.route('/delete_class/<cname>')
def delete_class(cname):
    db = get_database()
    admins = get_current_admin()
    session.pop('user', None)
    db.execute('delete from classroom where class_name  = ?', [cname])
    db.execute('delete from classroom where class_name = ?', [cname])
    db.commit()
    user_cursor = db.execute(' select c.class_name,sum(v.total_violations) from violations v, '
                             'classroom c where v.class_name = c.class_name group by c.class_name'
                             ' order by sum(v.total_violations)')
    data = user_cursor.fetchall()
    return render_template('classes.html', admins=admins, value=data)

@app.route('/admin_logout')
def admin_logout():
    error = None
    user = get_current_user()

    return render_template('admin_login.html', loginerror=error, user=user)


@app.route('/logging_out')
def logging_out():
    return redirect(url_for('login'))


@app.route('/report_user')
def report_user():
    user = get_current_user()
    db = get_database()
    class_cursor = db.execute('select * from classroom limit 4')
    class_data = class_cursor.fetchall()
    user_cursor = db.execute('select * from classroom')
    data = user_cursor.fetchall()

    return render_template('report_user.html', user=user, value=data)


@app.route('/generate_annual_report')
def generate_annual_report():
    user = get_current_user()
    return render_template('generate_annual_report.html', user=user)


@app.route('/report_data', methods=["POST", "GET"])
def report_data():
    global Class_Name
    global Report_Type
    global Report_Date
    global Recommendation
    if request.method == "POST":
        class_name = request.form['class_name']
        report_date = request.form['report_date']
        report_type = request.form['report']
        recommend = request.form['message']

        Class_Name = class_name
        Report_Type = report_type
        Report_Date = report_date
        Recommendation = recommend

    return redirect(url_for('pdf'))


@app.route('/generate_class_report')
def generate_class_report():
    global Class_Name
    global Report_Type
    global Report_Date
    global Recommendation
    db = get_database()
    user = get_current_user()
    if Report_Type == 'Annually':
        year_cursor = db.execute(
            ' select strftime("%y", start_of_detection),sum(v.social_violations),sum(v.violations),'
            'sum(v.total_violations),c.class_name from violations v, classroom c where v.class_name= ?'
            ' and v.class_name = c.class_name group by strftime("%y", start_of_detection)',
            [Class_Name])
        year_data = year_cursor.fetchall()
        result = year_data

        return render_template('generate_class_report.html', data=result, class_name=Class_Name,
                               report_date=Report_Date, report_type=Report_Type, recommend=Recommendation)

    elif Report_Type == 'Monthly':
        month_cursor = db.execute(
            'select CASE strftime("%m", start_of_detection) WHEN "01" THEN "January" WHEN "02" THEN "February" WHEN'
            ' "03" THEN "March" WHEN "04" THEN "+'
            'April" WHEN "05" THEN "May"WHEN "06" THEN "June" WHEN "07" THEN "July" WHEN "08" THEN "August" WHEN'
            ' "09" THEN "September" WHEN "10" THEN "October" WHEN "11" THEN "November" WHEN "12" THEN "December" '
            'END as month_name,sum(v.social_violations), sum(v.violations),sum(v.total_violations),c.class_name '
            ' from violations v, classroom c where v.class_name= ?'
            ' and v.class_name = c.class_name group by strftime("%m", start_of_detection)',
            [Class_Name])
        month_data = month_cursor.fetchall()
        result = month_data

        return render_template('generate_class_report.html', data=result, class_name=Class_Name,
                               report_date=Report_Date, report_type=Report_Type,recommend=Recommendation, user=user)
    elif Report_Type == 'Daily':
        day_cursor = db.execute(
            ' select strftime("%d-%m", start_of_detection),sum(v.social_violations),sum(v.violations),'
            'sum(v.total_violations),c.class_name from violations v, classroom c where v.class_name= ?'
            ' and v.class_name = c.class_name group by strftime("%d-%m", start_of_detection)'
            'order by strftime("%m", start_of_detection) ASC',
            [Class_Name])
        day_data = day_cursor.fetchall()
        result = day_data
        return render_template('generate_class_report.html', data=result, class_name=Class_Name,
                               report_date=Report_Date, report_type=Report_Type,recommend=Recommendation, user=user)

    return redirect(url_for('report_user'))


@app.route('/pdf')
def pdf():
    # Render the HTML to a PDF
    pdf = pdfkit.from_url("http://127.0.0.1:5000/generate_class_report", 'class.pdf')
    flash("Successfully Generated a class report", "success")
    return redirect(url_for('room'))


@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/social_d')
def social_d():
    return Response(generate_social(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(debug=True)

