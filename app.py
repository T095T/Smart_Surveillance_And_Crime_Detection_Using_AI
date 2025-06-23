from flask import (
    Flask,
    render_template,
    Response,
    request,
    redirect,
    url_for,
    jsonify,
    session,
)
import cv2
import numpy as np
import os
import time
from keras.models import load_model
from ultralytics import YOLO
from threading import Lock, Thread
from insightface.app import FaceAnalysis
import torch

app = Flask(__name__)
app.secret_key = "your_secret_key"
app.config["UPLOAD_FOLDER"] = os.path.join(os.getcwd(), "uploads")
if not os.path.exists(app.config["UPLOAD_FOLDER"]):
    os.makedirs(app.config["UPLOAD_FOLDER"])

# === Models and Setup ===
crime_model = load_model("Z:\\BE-PROJECT\\modelnew.h5")
crime_model = crime_model.cuda() if torch.cuda.is_available() else crime_model

weapon_model = YOLO("Z:\\BE-PROJECT\\best.pt")
weapon_model.eval()
if torch.cuda.is_available():
    weapon_model.to("cuda")

provider = (
    "CUDAExecutionProvider" if os.environ.get("CUDA_PATH") else "CPUExecutionProvider"
)
face_app = FaceAnalysis(name="buffalo_l", providers=[provider])
face_app.prepare(ctx_id=0 if provider == "CUDAExecutionProvider" else -1)
face_app.det_model.detect_size = (320, 320)

# === Criminal DB ===
criminal_db = {}
database_path = "../criminal-system/criminal_images"


def extract_embeddings(img):
    if img is None:
        return []
    img = cv2.resize(img, (480, 320))
    faces = face_app.get(img)
    if not faces:
        return []
    return [(face.embedding, face.bbox) for face in faces]


for filename in os.listdir(database_path):
    if filename.endswith((".jpg", ".png")):
        name = os.path.splitext(filename)[0]
        img = cv2.imread(os.path.join(database_path, filename))
        emb = extract_embeddings(img)
        if emb:
            criminal_db[name] = emb[0][0]


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def identify_criminal(frame, threshold=0.3):
    embeddings = extract_embeddings(frame)
    results = []
    for emb, bbox in embeddings:
        best_match = None
        best_score = -1
        for name, db_emb in criminal_db.items():
            similarity = cosine_similarity(emb, db_emb)
            if similarity > best_score:
                best_score = similarity
                best_match = name
        if best_score >= threshold:
            results.append(
                {"name": best_match, "score": best_score, "box": list(map(int, bbox))}
            )
    return results


def predict_crime(frame):
    frame_resized = cv2.resize(frame, (128, 128))
    frame_normalized = frame_resized / 255.0
    frame_input = np.expand_dims(frame_normalized, axis=0)
    prediction = crime_model.predict(frame_input)
    return prediction[0][0] > 0.5


def predict_weapon(frame, model):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model(frame_rgb, verbose=False)
    weapon_detected = False
    weapon_boxes = []

    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = model.names[cls_id].lower()

            if conf > 0.5 and ("weapon" in label or "gun" in label or "knife" in label):
                weapon_detected = True
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                weapon_boxes.append([x1, y1, x2 - x1, y2 - y1])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    label.capitalize(),
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )
    return weapon_detected, weapon_boxes


# === Alerts ===
alert_message = ""
alert_lock = Lock()


class VideoCamera:
    def __init__(self, source=0):
        self.cap = cv2.VideoCapture(source)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.lock = Lock()
        self.frame = None
        self.stopped = False
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        if not self.fps or self.fps <= 0:
            self.fps = 30  # fallback to 30 FPS if camera doesn't report FPS properly
        self.frame_delay = 1 / self.fps
        Thread(target=self.update, daemon=True).start()

    def update(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            if not ret:
                # Restart video if file (looping)
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            with self.lock:
                self.frame = cv2.resize(frame, (640, 480))

    def read(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def release(self):
        self.stopped = True
        self.cap.release()


camera = None


@app.route("/")
def index():
    video_mode = session.get("video_mode", "live")
    return render_template("index.html", video_mode=video_mode)


@app.route("/start", methods=["POST"])
def start():
    mode = request.form.get("mode")
    global alert_message

    with alert_lock:
        alert_message = {"type": "source", "message": "Source switched"}

    if mode == "video":
        video = request.files.get("videoFile")
        if video and video.filename != "":
            video_path = os.path.join(app.config["UPLOAD_FOLDER"], video.filename)
            video.save(video_path)
            session["video_mode"] = "video"
            session["video_path"] = video_path
            return redirect(url_for("video_feed"))
        else:
            session["video_mode"] = "live"
            return redirect(url_for("index", alert="Please upload a video file."))
    else:
        session["video_mode"] = "live"
        return redirect(url_for("video_feed"))


# seconds


def get_motion_box(prev_frame, current_frame, threshold=25):
    if prev_frame is None or current_frame is None:
        return None

    gray_prev = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    gray_curr = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gray_prev, gray_curr)
    _, thresh = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, None, iterations=2)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) > 500]
    return boxes


def generate_frames(camera):
    frame_count = 0
    violence_counter = 0
    start_time = None
    global alert_message
    last_alert_time = 0
    alert_cooldown = 10
    prev_frame = None
    weapon_counter = 0
    weapon_start_time = None


    while True:
        frame_start_time = time.time()

        frame = camera.read()
        if frame is None:
            time.sleep(0.01)
            continue

        frame_count += 1
        alert_triggered = False
        local_alert = ""
        alert_type = None  # NEW: track alert type explicitly

        motion_boxes = get_motion_box(prev_frame, frame)
        prev_frame = frame.copy()

        # Run detection every ~0.5 seconds (adjust as needed)
        if frame_count % int(camera.fps / 2) == 0:
            # === Crime Detection ===
            violence_detected = predict_crime(frame)
            if violence_detected and alert_type is None:
                if start_time is None:
                    start_time = time.time()
                violence_counter += 1
                if violence_counter >= 2 and (time.time() - start_time) <= 3:
                    local_alert = "🚨 Crime detected (violence)"
                    alert_type = "violence"
                    alert_triggered = True
                    violence_counter = 0
                    start_time = None
                elif (time.time() - start_time) > 5:
                    violence_counter = 0
                    start_time = None

            # Weapon detection with delay threshold
            weapon_detected, _ = predict_weapon(frame, weapon_model)

            if weapon_detected:
                if weapon_start_time is None:
                    weapon_start_time = time.time()
                weapon_counter += 1

                if weapon_counter >= 3 and (time.time() - weapon_start_time) <= 4 and alert_type is None:
                    local_alert = "🚨 Weapon detected"
                    alert_type = "weapon"
                    alert_triggered = True
                    weapon_counter = 0
                    weapon_start_time = None
                elif (time.time() - weapon_start_time) > 5:
                    weapon_counter = 0
                    weapon_start_time = None
            else:
                weapon_counter = 0
                weapon_start_time = None

            # Criminal detection
            criminals = identify_criminal(frame)
            for criminal in criminals:
                if alert_type is None:  # Only overwrite alert if none yet
                    x, y, x2, y2 = criminal["box"]
                    name = criminal["name"]
                    cv2.rectangle(frame, (x, y), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(
                        frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2
                    )
                    local_alert = f"🚨 Criminal Detected: {name}"
                    alert_type = "criminal"
                    alert_triggered = True
                    print(f"Criminal Detected: {name}")
                    break  # alert for only first detected criminal in frame

        # Send alert if triggered & cooldown passed
        if alert_triggered and time.time() - last_alert_time > alert_cooldown:
            with alert_lock:
                alert_message = {
                    "type": alert_type if alert_type else "unknown",
                    "message": local_alert,
                }
            anomaly_frame = frame.copy()  # Copy exact anomaly frame here
            send_telegram_alert(local_alert)
            send_telegram_photo(anomaly_frame, caption=local_alert)
            last_alert_time = time.time()

        ret, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
        frame_bytes = buffer.tobytes()
        yield (
            b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
        )

        # Calculate elapsed time for processing
        elapsed = time.time() - frame_start_time
        # Calculate remaining delay to maintain target FPS
        delay = camera.frame_delay - elapsed

        if delay > 0:
            time.sleep(delay)

        



import requests
from datetime import datetime

TELEGRAM_BOT_TOKEN = "7623141796:AAFr7D9cYjZWcEXp3FHXJzwnD5ZQ1QpDOgU"
TELEGRAM_CHAT_ID = "7530603091"


def send_telegram_photo(frame, caption=""):
    _, buffer = cv2.imencode(".jpg", frame)
    files = {"photo": ("image.jpg", buffer.tobytes(), "image/jpeg")}
    data = {"chat_id": TELEGRAM_CHAT_ID, "caption": caption}
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"

    try:
        requests.post(url, data=data, files=files)
    except requests.exceptions.RequestException as e:
        print(f"Telegram image alert failed: {e}")


def send_telegram_alert(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    full_message = f"{message}\n🕒 {timestamp}"
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    data = {"chat_id": TELEGRAM_CHAT_ID, "text": full_message}
    try:
        response = requests.post(url, data=data)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Telegram alert failed: {e}")


def save_clip(frames, path="alert_clip.mp4", fps=10):
    if not frames:
        return
    height, width, _ = frames[0].shape
    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    for frame in frames:
        out.write(frame)
    out.release()


def send_telegram_video(file_path, caption=""):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendVideo"
    try:
        with open(file_path, "rb") as video:
            files = {"video": video}
            data = {"chat_id": TELEGRAM_CHAT_ID, "caption": caption}
            requests.post(url, data=data, files=files)
    except Exception as e:
        print(f"Failed to send video: {e}")


# Keep track of current source outside request
current_source = None

@app.route("/video_feed")
def video_feed():
    global camera, current_source
    video_mode = session.get("video_mode")
    video_path = session.get("video_path")

    source = video_path if (video_mode == "video" and video_path) else 0

    if camera is None or current_source != source:
        if camera:
            camera.release()
        camera = VideoCamera(source=source)
        current_source = source

    return Response(generate_frames(camera), mimetype="multipart/x-mixed-replace; boundary=frame")



@app.route("/get_alert")
def get_alert():
    global alert_message
    with alert_lock:
        return jsonify({"alert_message": alert_message})


# if __name__ == '__main__':
#     app.run(debug=True, threaded=True)
