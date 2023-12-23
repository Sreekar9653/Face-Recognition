import pickle
import face_recognition
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import os

# Load known faces from the pickle file
with open("../model/pictureset.pickle", "rb") as filename:
    people = pickle.load(filename)

# Streamlit UI
st.title("Face Recognition App")

# Upload image through Streamlit file uploader
uploaded_file = st.file_uploader(
    "Choose a photo to recognize faces", type=["jpg", "jpeg"]
)

if uploaded_file is not None:
    # Load the image using face_recognition
    pic = face_recognition.load_image_file(uploaded_file)
    pic_coords = face_recognition.face_locations(pic, model="hog")
    pic_enc = face_recognition.face_encodings(pic, known_face_locations=pic_coords)
    face_pic = Image.fromarray(pic)

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
        draw = ImageDraw.Draw(face_pic)
        font = ImageFont.load_default()

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

    # Display the image with recognized faces
    st.image(face_pic, caption="Recognized Faces", use_column_width=True)

    # Learning Phase
    if len(unknown_faces_enc) > 0:
        st.write(
            f"There is(are) {len(unknown_faces_enc)} unknown person(s) in the photo."
        )
        if st.button("Enter their information"):
            st.write(
                "In each stage, enter an empty string if you do not know the person"
            )
            for i in range(0, len(unknown_faces_location)):
                top, right, bottom, left = unknown_faces_location[i]
                roi = face_pic.copy().crop([left, top, right, bottom])

                # Display the region of interest (ROI)
                st.image(roi, caption="Region of Interest", use_column_width=True)

                # Get user input for the person's name
                name = st.text_input("Who is this person?")

                if name in people:
                    tmp = people[name]
                    tmp.append(unknown_faces_enc[i])
                    st.write(
                        "The person was in the database. New photo was added to their profile."
                    )
                elif name:
                    people[name] = [unknown_faces_enc[i]]
                    try:
                        os.mkdir("People/" + name)
                    except:
                        pass
                    roi.save(
                        os.getcwd() + "/People/" + name + "/" + name + ".jpeg", "JPEG"
                    )
                    st.write("New person added")
                else:
                    st.write("Person skipped")

            # Save the updated database
            if st.button("Save Information"):
                with open("../model/pictureset.pickle", "wb") as filename:
                    pickle.dump(people, filename)
                st.write("Information saved successfully!")
