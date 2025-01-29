from face_recognition import register_user, recognize_face

def main():
    print("Welcome to the Face Recognition Attendance System")
    action = input("Enter 'register' to register a new user or 'recognize' to start face recognition: ").strip().lower()

    if action == "register":
        user_id = input("Enter your user ID: ").strip()
        register_user(user_id)
    elif action == "recognize":
        recognize_face()
    else:
        print("Invalid action. Please enter 'register' or 'recognize'.")

if __name__ == "__main__":
    main()
