import os
import cv2
import json
import pandas as pd
import numpy as np
from datetime import datetime
from mtcnn.mtcnn import MTCNN  # Import MTCNN for face detection

FACE_DATASET_PATH = "OneDrive/FaceImages"
ATTENDANCE_LOG_PATH = "OneDrive/Logs/CheckInLogs.csv"
USER_METADATA_PATH = "Metadata/UserDetails.json"

def initialize_users_file():
    if not os.path.exists(USER_METADATA_PATH):
        with open(USER_METADATA_PATH, 'w') as f:
            json.dump({}, f)

def generate_employee_id():
    # Load existing users to get the latest employee ID
    if os.path.exists(USER_METADATA_PATH):
        with open(USER_METADATA_PATH, "r") as f:
            users = json.load(f)
            return len(users) + 1  # Generate a new employee ID based on the existing records
    return 1  # If no users, start from ID 1

def register_user():
    user_name = input("Enter user's name: ").strip()
    mobile_no = input("Enter mobile number: ").strip()
    father_name = input("Enter father's name: ").strip()
    mother_name = input("Enter mother's name: ").strip()
    
    # Generate unique employee ID
    employee_id = generate_employee_id()
    print(f"Generated Employee ID: {employee_id}")

    # Capture face
    cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    detector = MTCNN()  # Initialize the MTCNN detector
    
    print(f"Starting face capture for user: {user_name} (Employee ID: {employee_id})")
    
    initialize_users_file()

    while True:
        ret, frame = cam.read()
        if not ret:
            print("❌ Failed to capture frame. Check your camera.")
            break

        # Detect faces using MTCNN
        faces = detector.detect_faces(frame)

        cv2.putText(frame, "Press 'C' to capture face", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        for face in faces:
            x, y, w, h = face['box']
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.imshow("Face Capture", frame)

            if cv2.waitKey(1) & 0xFF == ord('c'):
                face_img = frame[y:y + h, x:x + w]
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                file_path = os.path.join(FACE_DATASET_PATH, f"{employee_id}_{timestamp}.jpg")
                cv2.imwrite(file_path, face_img)
                print(f"✅ Face saved for {user_name} (Employee ID: {employee_id})")

                # Save user details in the metadata file
                with open(USER_METADATA_PATH, "r") as f:
                    users = json.load(f)

                users[str(employee_id)] = {
                    'name': user_name,
                    'mobile_no': mobile_no,
                    'father_name': father_name,
                    'mother_name': mother_name,
                    'image_path': file_path
                }

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

def recognize_face():
    cam = cv2.VideoCapture(0)
    users = {}

    if os.path.exists(USER_METADATA_PATH):
        with open(USER_METADATA_PATH, "r") as f:
            users = json.load(f)

    detector = MTCNN()  # Initialize the MTCNN detector
    known_faces = {}

    for employee_id, data in users.items():
        img = cv2.imread(data['image_path'])
        faces = detector.detect_faces(img)
        if faces:
            # Use the first detected face's bounding box and keypoints for recognition
            x, y, w, h = faces[0]['box']
            face_img = img[y:y + h, x:x + w]
            known_faces[employee_id] = face_img

    print("🔍 Start recognizing faces...")
    while True:
        ret, frame = cam.read()
        if not ret:
            print("❌ Camera not detected.")
            break

        # Detect faces using MTCNN
        faces = detector.detect_faces(frame)

        for face in faces:
            x, y, w, h = face['box']
            face_img = frame[y:y + h, x:x + w]
            
            # Compare faces
            best_match_id = "Unknown"
            best_match_score = float('inf')  # Initialize to a very high score
            
            for employee_id, known_face in known_faces.items():
                # Calculate the difference between the captured face and the known face
                diff = np.sum(np.abs(known_face - face_img))
                if diff < best_match_score:
                    best_match_score = diff
                    best_match_id = employee_id

            # If the match score is below a certain threshold, consider it a match
            if best_match_score < 1000000:  # You can adjust the threshold
                name = users[best_match_id]['name']
                print(f"✅ Face recognized: {name}")
            else:
                name = "Unknown"

            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.imshow("Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()
