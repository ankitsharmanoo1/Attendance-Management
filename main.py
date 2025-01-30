from face_recognition import register_user, recognize_face

def main():
    print("Welcome to the Face Recognition Attendance System")
    action = input("Enter 'register' to register a new user or 'recognize' to start face recognition: ").strip().lower()

    if action == "register":
        register_user()  # No need to pass user_id here
    elif action == "recognize":
        recognize_face()
    else:
        print("Invalid action. Please enter 'register' or 'recognize'.")

if __name__ == "__main__":
    main()
