import cv2
import pickle
import face_recognition
import PIL
import PIL.ImageFont
import PIL.ImageDraw
import numpy as np
font = PIL.ImageFont.truetype("timesbd.ttf", 20)

with open('pictureset.pickle', 'rb') as filename:
    people = pickle.load(filename)

cap = cv2.VideoCapture(0)
while(cap.isOpened()):
    ret, frame = cap.read()
    small_frame = cv2.resize(frame, (0, 0), fx=1/4, fy=1/4)
    rgb_small_frame = small_frame[:,:,::-1]
    img_loc = face_recognition.face_locations(rgb_small_frame,model="hog")
    img_enc = face_recognition.face_encodings(rgb_small_frame, known_face_locations=img_loc)

    face_img = PIL.Image.fromarray(frame)

    for i in range(0, len(img_enc)):
        name="unknown"
        for k, v in people.items():
            result = face_recognition.compare_faces(v, img_enc[i], tolerance=0.5)
            top, right, bottom, left = np.multiply(img_loc[i], 4)
            draw = PIL.ImageDraw.Draw(face_img)
            draw.rectangle([left, top, right, bottom], outline="red", width=2)
            if True in result:
                name=k
        draw.text((left, bottom), name, font=font)

    open_cv_image = np.array(face_img)
    cv2.imshow('frame',open_cv_image.copy())
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
