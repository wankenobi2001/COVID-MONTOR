create table users(id integer AUTO_INCREMENT PRIMARY KEY, Username text not null, password text not null,
admin boolean not null DEFAULT '0');

create table employee( empid integer AUTO_INCREMENT PRIMARY KEY, fname text not null, lname text not null,
 Username text not null, email text not null, phone text not null, nationalID text not null, joining_date timestamp DEFAULT CURRENT_TIMESTAMP);