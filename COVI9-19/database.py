from flask import g
import sqlite3

def connect_to_database():
    sql = sqlite3.connect('C:\wamp64\www\COVI9-19\COVID.db')
    sql.row_factory = sqlite3.Row
    return sql

def get_database():
    if not hasattr(g, 'COVID_db'):
        g.COVID_db = connect_to_database()

    return g.COVID_db





