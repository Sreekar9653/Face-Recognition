import streamlit as st
import cv2
import pickle
import face_recognition
import PIL
import PIL.ImageFont
import PIL.ImageDraw
import numpy as np
import time

font = PIL.ImageFont.truetype("timesbd.ttf", 20)

with open("./model/pictureset.pickle", "rb") as filename:
    people = pickle.load(filename)

st.title("Face Recognition")

# Initialize variables
start_button = st.button("Start Face Recognition")
stop_button = st.button("Stop Face Recognition")
counter = 0  # Counter for generating unique keys

# Create an image container
image_container = st.empty()

# OpenCV video capture
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 60)  # Adjust the value according to your needs

while True:
    if start_button:
        ret, frame = cap.read()

        if not ret:
            break

        small_frame = cv2.resize(frame, (0, 0), fx=1 / 4, fy=1 / 4)
        rgb_small_frame = small_frame[:, :, ::-1]
        img_loc = face_recognition.face_locations(rgb_small_frame, model="hog")
        img_enc = face_recognition.face_encodings(
            rgb_small_frame, known_face_locations=img_loc
        )

        face_img = PIL.Image.fromarray(frame)

        for i in range(0, len(img_enc)):
            name = "unknown"
            for k, v in people.items():
                result = face_recognition.compare_faces(v, img_enc[i], tolerance=0.5)
                top, right, bottom, left = np.multiply(img_loc[i], 4)
                draw = PIL.ImageDraw.Draw(face_img)
                draw.rectangle([left, top, right, bottom], outline="red", width=2)
                if True in result:
                    name = k
            draw.text((left, bottom), name, font=font)

        open_cv_image = np.array(face_img)

        # Display the camera feed
        image_container.image(
            open_cv_image, channels="BGR", use_column_width=True, caption="Camera Feed"
        )

        # Introduce a delay to control the frame rate
        time.sleep(0.03)  # Adjust the value according to your needs

        counter += 1  # Increment the counter for the next iteration

    elif stop_button:
        # Break the loop when the stop button is pressed
        break

# Release the video capture when done
cap.release()
