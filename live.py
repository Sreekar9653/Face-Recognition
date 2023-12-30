from flask import Flask, render_template, Response, request
import cv2
import pickle
import face_recognition
import PIL
import PIL.ImageFont
import PIL.ImageDraw
import numpy as np
import time
import json

app = Flask(__name__)

font = PIL.ImageFont.truetype("timesbd.ttf", 20)

attendance_list = list()
json_file_path = f"./attendances/attendance_{time.strftime('%Y-%m-%d')}.json"


def save_attendance_to_json():
    global attendance_list
    with open(json_file_path, "w") as json_file:
        json.dump(attendance_list, json_file, indent=2)


def process_frame(frame):
    global attendance_list

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

    return np.array(face_img)


def generate_frames():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        return

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        processed_frame = process_frame(frame)
        _, buffer = cv2.imencode(".jpg", processed_frame)
        frame = buffer.tobytes()

        yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/attendance_info")
def get_attendance_info():
    return json.dumps(attendance_list)


@app.route("/video_feed")
def video_feed():
    return Response(
        generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/save_attendance", methods=["POST"])
def save_attendance():
    # Call the function to save attendance to JSON file
    save_attendance_to_json()
    return "Attendance saved successfully!"


if __name__ == "__main__":
    app.run(debug=True)
