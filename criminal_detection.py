import os
import cv2
import numpy as np
from insightface.app import FaceAnalysis

def initialize_face_app():
    """Initialize InsightFace FaceAnalysis app with correct provider."""
    provider = "CUDAExecutionProvider" if os.environ.get("CUDA_PATH") else "CPUExecutionProvider"
    app = FaceAnalysis(name="buffalo_l", providers=[provider])
    app.prepare(ctx_id=0 if provider == "CUDAExecutionProvider" else -1)
    app.det_model.detect_size = (320, 320)  # smaller size for speed
    return app

def extract_embeddings(app, img):
    """Extract embeddings and bounding boxes for all faces in an image/frame."""
    if img is None:
        return []
    img_resized = cv2.resize(img, (640, 480))  # Resize to reduce computation
    faces = app.get(img_resized)
    if not faces:
        return []
    return [(face.embedding, face.bbox) for face in faces]

def build_criminal_database(app, database_path= r"Z:\BE-PROJECT-final - This has less laggy cam-feed\criminal-system\criminal_images"):
    """Load criminal face embeddings from images in database_path."""
    criminal_db = {}
    for filename in os.listdir(database_path):
        if filename.lower().endswith((".jpg", ".png")):
            person_name = os.path.splitext(filename)[0]
            img = cv2.imread(os.path.join(database_path, filename))
            embeddings = extract_embeddings(app, img)
            if embeddings:  # Take first detected face
                criminal_db[person_name] = embeddings[0][0]
    return criminal_db

def cosine_similarity(a, b):
    """Calculate cosine similarity between two embeddings."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def identify_criminal(app, criminal_db, frame, threshold=0.5):
    """
    Identify faces in a frame using criminal_db embeddings.

    Returns:
      - results: list of tuples (match_name or None, similarity_score, bbox)
      - face_detected: bool indicating if any face was detected
    """
    embeddings = extract_embeddings(app, frame)
    if not embeddings:
        return [], False  # No faces detected

    results = []
    for embedding, bbox in embeddings:
        best_match = None
        best_score = -1
        for name, db_emb in criminal_db.items():
            sim = cosine_similarity(embedding, db_emb)
            if sim > best_score:
                best_score = sim
                best_match = name
        if best_score >= threshold:
            results.append((best_match, best_score, bbox))
        else:
            results.append((None, best_score, bbox))  # Face detected, no match
    return results, True

def identify_criminal_dict(app, criminal_db, frame, threshold=0.5):
    """
    Same as identify_criminal but returns list of dicts with keys: 'match', 'score', 'box'.
    Useful if you want to use keys to access bbox in external code.
    """
    results, face_detected = identify_criminal(app, criminal_db, frame, threshold)
    results_dicts = []
    for match, score, bbox in results:
        results_dicts.append({
            "match": match,
            "score": score,
            "box": bbox.tolist() if hasattr(bbox, 'tolist') else bbox
        })
    return results_dicts, face_detected
