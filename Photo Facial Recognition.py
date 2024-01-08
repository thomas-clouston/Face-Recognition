# Photo Facial Recognition


# Imports
import face_recognition
import os
import cv2


# Variables
unknown_faces_dir = 'unknown_faces'
known_faces_dir = 'known_faces'
frame_thickness = 3
font_thickness = 2
model_type = 'hog'
box_colour = 0, 255, 0
text_colour = 0, 0, 0


known_faces = []
known_names = []


# Organise faces
for name in os.listdir(known_faces_dir):

    # Next we load every file of faces of known person
    for filename in os.listdir(f'{known_faces_dir}/{name}'):

        # Load image
        image = face_recognition.load_image_file(f'{known_faces_dir}/{name}/{filename}')

        # Encode image
        encoding = face_recognition.face_encodings(image)[0]

        # Append encodings and name
        known_faces.append(encoding)
        known_names.append(name)

# Now let's loop over a folder of faces we want to label
for filename in os.listdir(unknown_faces_dir):

    # Load image
    image = face_recognition.load_image_file(f'{unknown_faces_dir}/{filename}')

    # Find face locations
    locations = face_recognition.face_locations(image, model=model_type)

    # Encode image
    encodings = face_recognition.face_encodings(image, locations)

    # Convert from RGB to BGR
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    for face_encoding, face_location in zip(encodings, locations):

        # Check for match
        results = face_recognition.compare_faces(known_faces, face_encoding)

        match = None
        if True in results:  # If at least one is true, get a name of first of found labels
            match = known_names[results.index(True)]

            # Find top face positions
            top_left = (face_location[3], face_location[0])
            bottom_right = (face_location[1], face_location[2])

            # Paint top frame
            cv2.rectangle(image, top_left, bottom_right, box_colour, frame_thickness)

            # Find bottom face positions
            top_left = (face_location[3], face_location[2])
            bottom_right = (face_location[1], face_location[2] + 22)

            # Paint bottom frame
            cv2.rectangle(image, top_left, bottom_right, box_colour, cv2.FILLED)

            # Paint name
            cv2.putText(image, match, (face_location[3] + 10, face_location[2] + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_colour, font_thickness)

    # Show image
    cv2.imshow(filename, image)
    cv2.waitKey(0)
    cv2.destroyWindow(filename)
