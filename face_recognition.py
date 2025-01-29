import os
import cv2
import json
import numpy as np
import pandas as pd
import face_recognition
from config import FACE_DATASET_PATH, ATTENDANCE_LOG_PATH, USER_METADATA_PATH

# Create necessary folders if they don't exist
os.makedirs(FACE_DATASET_PATH, exist_ok=True)
os.makedirs(os.path.dirname(ATTENDANCE_LOG_PATH), exist_ok=True)
os.makedirs(os.path.dirname(USER_METADATA_PATH), exist_ok=True)

# Function to initialize user metadata file if it doesn't exist
def initialize_users_file():
    if not os.path.exists(USER_METADATA_PATH):
        with open(USER_METADATA_PATH, 'w') as f:
            json.dump({}, f)

# Function to register a user by capturing their face
def register_user(user_id):
    cam = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    
    print(f"Starting face capture for user: {user_id}")
    
    # Initialize the users file if it doesn't exist
    initialize_users_file()

    # Capture the face
    while True:
        ret, frame = cam.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        for (x, y, w, h) in faces:
            face_img = frame[y:y + h, x:x + w]
            file_path = os.path.join(FACE_DATASET_PATH, f"{user_id}.jpg")
            cv2.imwrite(file_path, face_img)
            print(f"Face saved for {user_id} at {file_path}")
            
            # Load existing users from the file and add the new user
            with open(USER_METADATA_PATH, "r") as f:
                users = json.load(f)

            # Add the new user to the dictionary
            users[user_id] = file_path

            # Save the updated user data to the file
            with open(USER_METADATA_PATH, "w") as f:
                json.dump(users, f)

            cam.release()
            cv2.destroyAllWindows()
            return

        cv2.imshow("Face Capture", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()

# Function to recognize faces and mark attendance
def recognize_face():
    cam = cv2.VideoCapture(0)
    
    # Load user face encodings
    users = {}
    if os.path.exists(USER_METADATA_PATH):
        with open(USER_METADATA_PATH, "r") as f:
            users = json.load(f)

    known_faces = {}
    for user_id, img_path in users.items():
        img = face_recognition.load_image_file(img_path)
        known_faces[user_id] = face_recognition.face_encodings(img)[0]

    print("Start recognizing faces...")
    while True:
        ret, frame = cam.read()
        rgb_frame = frame[:, :, ::-1]  # Convert BGR to RGB
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(list(known_faces.values()), face_encoding)
            name = "Unknown"
            
            # If a match is found, recognize the user
            if True in matches:
                first_match_index = matches.index(True)
                name = list(known_faces.keys())[first_match_index]
                print(f"✅ Face recognized: {name}")
                mark_attendance(name)
                
        cv2.imshow("Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()

# Function to mark attendance in a CSV file
def mark_attendance(user_id):
    now = pd.to_datetime("now")
    log = pd.read_csv(ATTENDANCE_LOG_PATH) if os.path.exists(ATTENDANCE_LOG_PATH) else pd.DataFrame(columns=["User", "Time"])

    # Check if the user has already marked attendance today
    if not log[log['User'] == user_id].empty:
        print(f"Attendance for {user_id} has already been marked.")
        return

    log = log.append({"User": user_id, "Time": now.strftime("%Y-%m-%d %H:%M:%S")}, ignore_index=True)
    log.to_csv(ATTENDANCE_LOG_PATH, index=False)
    print(f"✅ Attendance marked for {user_id} at {now}")
