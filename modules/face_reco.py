import pickle
import face_recognition
import PIL
import cv2
import PIL.ImageFont
import PIL.ImageDraw
import numpy as np
import time
import numpy as np

font = PIL.ImageFont.truetype("timesbd.ttf", 20)


def process_frame(frame, attendance_list):
    with open("./model/pictureset.pickle", "rb") as filename:
        people = pickle.load(filename)

    small_frame = cv2.resize(frame, (0, 0), fx=1 / 4, fy=1 / 4)
    rgb_small_frame = small_frame[:, :, ::-1]
    img_loc = face_recognition.face_locations(rgb_small_frame, model="hog")
    img_enc = face_recognition.face_encodings(
        rgb_small_frame, known_face_locations=img_loc
    )

    face_img = PIL.Image.fromarray(frame)

    for i in range(0, len(img_enc)):
        name = "Unknown"
        for k, v in people.items():
            result = face_recognition.compare_faces(v, img_enc[i], tolerance=0.5)

            top, right, bottom, left = np.multiply(img_loc[i], 4)
            draw = PIL.ImageDraw.Draw(face_img)
            draw.rectangle([left, top, right, bottom], outline="red", width=2)

            if True in result:
                name = k
                draw.text((left, bottom), name, font=font)
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                attendee = {"name": name, "timestamp": timestamp, "present": True}
                if not any(
                    existing_attendee["name"] == attendee["name"]
                    for existing_attendee in attendance_list
                ):
                    attendance_list.append(attendee)

    return (np.array(face_img), attendance_list)
