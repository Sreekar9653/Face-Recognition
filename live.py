from flask import Flask, render_template, Response, request
import cv2
import time
import json

from modules.face_reco import process_frame

app = Flask(__name__)


attendance_list = list()

__JSON_FILE_PATH__ = f"./attendances/attendance_{time.strftime('%Y-%m-%d')}.json"

# ?global variable to control video feed
video_feed_active = True
process_framedata = False


def save_attendance_to_json():
    global attendance_list
    with open(__JSON_FILE_PATH__, "w") as json_file:
        json.dump(attendance_list, json_file, indent=2)


def generate_frames():
    global attendance_list

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        return

    global video_feed_active, process_framedata

    while video_feed_active:
        ret, frame = cap.read()

        if not ret:
            break

        if process_framedata:
            processed_frame, attendance_list = process_frame(frame, attendance_list)
            _, buffer = cv2.imencode(".jpg", processed_frame)
        else:
            _, buffer = cv2.imencode(".jpg", frame)

        frame_bytes = buffer.tobytes()
        yield (
            b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
        )

    # Release the camera when the video feed is no longer active
    cap.release()


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


@app.route("/startAttendance", methods=["POST"])
def start_attendance():
    global process_framedata
    process_framedata = True
    return "Attendance Started successfully!"


@app.route("/remove_attendee", methods=["POST"])
def remove_attendee():
    global attendance_list

    # Get the name to be removed from the request JSON data
    data = request.get_json()
    name_to_remove = data.get("name", "")

    print(f"Remove {name_to_remove}")
    # Remove the attendee with the specified name
    attendance_list = [
        attendee for attendee in attendance_list if attendee["name"] != name_to_remove
    ]

    return "Attendee removed successfully!"


@app.route("/clearAttendance", methods=["POST"])
def clear_attendance():
    global attendance_list
    attendance_list.clear()
    return "Attendance cleared successfully!"


# Add a route to stop the video feed
@app.route("/stopVideoFeed", methods=["POST"])
def stop_video_feed():
    global process_framedata
    process_framedata = False
    print("Frame Else Part")
    return "Attendence Stopped"


if __name__ == "__main__":
    app.run(debug=True)
