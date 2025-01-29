import os

# Define paths to the required directories and files
BASE_PATH = os.path.dirname(os.path.realpath(__file__))

FACE_DATASET_PATH = os.path.join(BASE_PATH, "FaceImages")
ATTENDANCE_LOG_PATH = os.path.join(BASE_PATH, "Logs", "CheckInLogs.csv")
USER_METADATA_PATH = os.path.join(BASE_PATH, "Metadata", "UserDetails.json")
