# Webcam Facial Recognition


# Imports
import face_recognition
import os
import cv2


# Parameters
known_faces_dir = 'known_faces'
frame_thickness = 3
font_thickness = 2
model_type = 'hog'
box_colour = 0, 255, 0
text_colour = 0, 0, 0
video = cv2.VideoCapture(0)


# Lists
known_faces = []
known_names = []


# Image organisation
for name in os.listdir(known_faces_dir):

    # Load every file of faces of known person
    for filename in os.listdir(f'{known_faces_dir}/{name}'):

        # Load face images
        image = face_recognition.load_image_file(f'{known_faces_dir}/{name}/{filename}')

        # Build 128-dimension face encoding
        encoding = face_recognition.face_encodings(image)[0]

        # Append encodings and name
        known_faces.append(encoding)
        known_names.append(name)


# Facial recognition
while True:
    # Load video
    ret, frame = video.read()

    # Get face locations
    locations = face_recognition.face_locations(frame, model=model_type)

    # Level-2 encoding
    encodings = face_recognition.face_encodings(frame, locations)

    # Finding faces
    for face_encoding, face_location in zip(encodings, locations):

        # Comparing faces
        results = face_recognition.compare_faces(known_faces, face_encoding)

        # Returning found faces
        if True in results:
            match = known_names[results.index(True)]

            # Getting face positions for box
            top_left = (face_location[3], face_location[0])
            bottom_right = (face_location[1], face_location[2])

            # Paint box frame
            cv2.rectangle(frame, top_left, bottom_right, box_colour, frame_thickness)

            # Getting face positions for text
            top_left = (face_location[3], face_location[2])
            bottom_right = (face_location[1], face_location[2] + 22)

            # Paint text frame
            cv2.rectangle(frame, top_left, bottom_right, box_colour, cv2.FILLED)

            # Write name
            cv2.putText(frame, match, (face_location[3] + 10, face_location[2] + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        text_colour, font_thickness)

    # Show video
    cv2.imshow('Webcam', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
