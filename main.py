from flask import Flask, render_template, request, Response, jsonify
from predict import predict_pose 
import cv2
import os
import numpy as np
from flask import send_from_directory

import tensorflow as tf
import mediapipe as mp
import time

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

model_path = os.path.join(os.getcwd(), "yoga_pose_model.h5")
model = tf.keras.models.load_model(model_path)

mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils
pose = mp_pose.Pose()

feedback_text = "Waiting for feedback..."
last_feedback = ""
last_feedback_time = time.time()


def preprocess_image(image):
    """Preprocess the input image for model prediction."""
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224)) 
    image = image / 255.0  
    image = np.expand_dims(image, axis=0) 
    return image

last_feedback = None
last_feedback_time = 0

def calculate_angle(a, b, c):
    """Calculate the angle between three points: a (joint1), b (joint2), c (joint3)."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    
    ba = a - b
    bc = c - b

    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)
    
    if norm_ba == 0 or norm_bc == 0:
        return 0  
    cosine_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)  
    return np.degrees(np.arccos(cosine_angle))

def detect_goddess_pose(left_knee_angle, right_knee_angle):
    """Detect Goddess Pose"""
    return 85 <= left_knee_angle <= 95 and 85 <= right_knee_angle <= 95

def detect_warrior_pose(left_knee_angle, right_knee_angle):
    """Detect Warrior Pose"""
    return 85 <= left_knee_angle <= 95 and right_knee_angle > 160

def detect_plank_pose(left_knee_angle, right_knee_angle, hip_angle):
    """Detect Plank Pose"""
    return left_knee_angle > 160 and right_knee_angle > 160 and hip_angle > 150

def detect_tree_pose(left_ankle, left_knee, right_ankle, right_knee):
    """Detect Tree Pose"""
    return left_ankle[1] < left_knee[1] and right_ankle[1] > right_knee[1]

def detect_downward_dog(hip_angle, left_knee_angle, right_knee_angle):
    """Detect Downward Dog Pose"""
    return hip_angle > 160 and left_knee_angle > 150 and right_knee_angle > 150

def analyze_pose(keypoints):
    """Analyze detected pose and provide specific feedback."""

    required_keypoints = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
    if not all(i in keypoints for i in required_keypoints):
        return "Stand straight and align your pose properly."

    left_shoulder, right_shoulder = keypoints[11], keypoints[12]
    left_elbow, right_elbow = keypoints[13], keypoints[14]
    left_wrist, right_wrist = keypoints[15], keypoints[16]
    left_hip, right_hip = keypoints[23], keypoints[24]
    left_knee, right_knee = keypoints[25], keypoints[26]
    left_ankle, right_ankle = keypoints[27], keypoints[28]

    left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
    right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)
    hip_angle = calculate_angle(left_shoulder, left_hip, left_knee)

    if detect_goddess_pose(left_knee_angle, right_knee_angle):
        feedback = "Goddess Pose Detected! Keep knees at 90° and arms strong."
    elif detect_warrior_pose(left_knee_angle, right_knee_angle):
        feedback = "Warrior Pose Detected! Keep your back leg straight and stretch forward."
    elif detect_plank_pose(left_knee_angle, right_knee_angle, hip_angle):
        feedback = "Plank Pose Detected! Engage your core and keep your body straight."
    elif detect_tree_pose(left_ankle, left_knee, right_ankle, right_knee):
        feedback = "Tree Pose Detected! Focus on balance and keep your hands together."
    elif detect_downward_dog(hip_angle, left_knee_angle, right_knee_angle):
        feedback = "Downward Dog Detected! Keep your hips high and heels pressed down."
    else:
        feedback = "Adjust your pose!"

    global last_feedback, last_feedback_time
    if feedback != last_feedback and time.time() - last_feedback_time > 2:
        last_feedback = feedback
        last_feedback_time = time.time()

    return last_feedback

def generate_frames():
    """Capture video, process pose, and stream frames."""
    global feedback_text

    cap = cv2.VideoCapture(0)

    try:
        while True:
            success, frame = cap.read()
            if not success:
                break

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)

            if results.pose_landmarks:
                mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                keypoints = {idx: (int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0]))
                             for idx, landmark in enumerate(results.pose_landmarks.landmark)}

                feedback_text = analyze_pose(keypoints)

            _, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    except Exception as e:
        print("Error:", e)

    finally:
        cap.release()


@app.route('/')
def home():
    return render_template('index.html')


@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return "No file uploaded", 400

    file = request.files["file"]
    
    if file.filename == "":
        return "No selected file", 400

    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

    filepath = os.path.join(app.config["UPLOAD_FOLDER"], 'latest_pose.jpg')
    file.save(filepath)

    predicted_pose = predict_pose(filepath)

    return render_template("result.html", pose=predicted_pose)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)





@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/get_feedback')
def get_feedback():
    global feedback_text
    return jsonify({"feedback": feedback_text})


if __name__ == "__main__":
    app.run(debug=True)
