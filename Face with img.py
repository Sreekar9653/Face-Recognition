import pickle
import face_recognition
import PIL
import PIL.Image
import PIL.ImageFont
import os


import PIL.ImageDraw
from tkinter import filedialog
from tkinter import *
import math
import warnings

warnings.filterwarnings("ignore")

with open("./model/pictureset.pickle", "rb") as filename:
    people = pickle.load(filename)

print("Select a file")
root = Tk()
root.filename = filedialog.askopenfilename(
    initialdir="/",
    title="Select photo to recognize faces",
    filetypes=(("jpeg files", "*.jpg"), ("all files", "*.*")),
)
print(root.filename)
root.destroy()

pic = face_recognition.load_image_file(root.filename)
pic_coords = face_recognition.face_locations(pic, model="hog")
pic_enc = face_recognition.face_encodings(pic, known_face_locations=pic_coords)
face_pic = PIL.Image.fromarray(pic)

unknown_faces_location = []
unknown_faces_enc = []
for i in range(0, len(pic_enc)):
    faces = 0
    persname = "unknown"
    for key, value in people.items():
        result = face_recognition.compare_faces(value, pic_enc[i], tolerance=0.5)
        faces_found = result.count(True)
        if faces_found > faces:
            faces = faces_found
            persname = key
    top, right, bottom, left = pic_coords[i]
    draw = PIL.ImageDraw.Draw(face_pic)
    font = PIL.ImageFont.load_default()

    text_bbox = draw.textbbox((left, bottom), persname, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]

    draw.rectangle(
        (left, bottom, left + text_width, bottom + text_height * 1.2), fill="black"
    )
    draw.text((left, bottom), persname, font=font)

    if faces == 0:
        unknown_faces_location.append(pic_coords[i])
        unknown_faces_enc.append(pic_enc[i])
face_pic.show()

# Learning Phase
if len(unknown_faces_enc) > 0:
    print(
        "There is(are)",
        len(unknown_faces_enc),
        "unknown person(s) in the photo. Would you like to enter their information? (Y|N)",
    )
    if input().lower() in ["y", "yes"]:
        print("In each stage, enter empty string if you do not know the person")
        for i in range(0, len(unknown_faces_location)):
            top, right, bottom, left = unknown_faces_location[i]
            roi = face_pic.copy().crop([left, top, right, bottom])
            roi.show()
            name = input("Who is this person? ")
            if name in people:
                tmp = people[name]
                tmp.append(unknown_faces_enc[i])
                print(
                    "The person was in the database. New photo was added to their profile."
                )
            elif name:
                people[name] = [unknown_faces_enc[i]]
                try:
                    os.mkdir("People/" + name)
                except:
                    pass
                roi.save(os.getcwd() + "/People/" + name + "/" + name + ".jpeg", "JPEG")
                print("New person added")
            else:
                print("Person skipped")
        with open("pictureset.pickle", "wb") as filename:
            pickle.dump(people, filename)
