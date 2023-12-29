import streamlit as st
import cv2
import pickle
import face_recognition
import PIL
import PIL.ImageFont
import PIL.ImageDraw
import numpy as np
import time
import json

font = PIL.ImageFont.truetype("timesbd.ttf", 20)

with open("./model/pictureset.pickle", "rb") as filename:
    people = pickle.load(filename)

st.title("Face Recognition Streamlit App")

# List to store attendance information as dictionaries
attendance_list = list()

# File to store attendance information in JSON format
json_file_path = "./attendance.json"


# Function to process each frame
def process_frame(frame):
    global attendance_list

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
                # Add attendance record to the list only if the name is not present
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                attendee = {"name": name, "timestamp": timestamp, "present": True}
                if not any(
                    existing_attendee["name"] == attendee["name"]
                    for existing_attendee in attendance_list
                ):
                    attendance_list.append(attendee)
                    # Save the updated attendance list to the JSON file
                    with open(json_file_path, "w") as json_file:
                        json.dump(attendance_list, json_file, indent=2)

                # st.text(attendance_list)
                # st.text(f"Count {len(attendance_list)}")
    return np.array(face_img)


# Main Streamlit app
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    st.error("Unable to open the camera. Please check your camera connection.")

start_stop_switch = st.checkbox("Start/Stop Attendance Collection")
stframe = st.empty()

while start_stop_switch:
    ret, frame = cap.read()

    if not ret:
        break

    processed_frame = process_frame(frame)

    stframe.image(processed_frame, channels="BGR", use_column_width=True)

    # Introduce a delay to control the frame rate
    time.sleep(0.03)  # Adjust the value according to your needs

# Print the attendance list when attendance collection stops
st.write("Attendance Information:")

cap.release()
cv2.destroyAllWindows()
