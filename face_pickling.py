import PIL.Image
import PIL.ImageDraw
import numpy
import pandas as pd
import tensorflow as tf
import face_recognition
import os
import pickle

people = {}
people_folder = os.getcwd() + "/people/"
print("Loading faces:")
for person in os.listdir(people_folder):
    print(person)
    tempfolder = people_folder + person + "/"
    person_faces = []
    for photo in os.listdir(tempfolder):
        image = face_recognition.load_image_file(tempfolder + photo)
        encode = face_recognition.face_encodings(image)
        if len(encode) > 0:
            person_faces.append(encode[0])
    if len(person_faces) != 0:
        people[person] = person_faces

with open("./model/pictureset.pickle", "wb") as filename:
    pickle.dump(people, filename)
