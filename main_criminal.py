# my_app.py
import cv2
from criminal_detector import initialize_face_app, build_criminal_database, identify_criminal_dict

def main():
    app = initialize_face_app()
    criminal_db = build_criminal_database(app, database_path="path/to/criminal_images")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        if frame_count % 5 == 0:
            results, detected = identify_criminal_dict(app, criminal_db, frame, threshold=0.3)
            if detected:
                for res in results:
                    x1, y1, x2, y2 = map(int, res["box"])
                    if res["match"]:
                        print(f"Detected {res['match']} with score {res['score']:.2f}")
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                        cv2.putText(frame, f"{res['match']} ({res['score']:.2f})", (x1, y1-10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                    else:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,0,255), 2)
                        cv2.putText(frame, "No match", (x1, y1-10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
            else:
                cv2.putText(frame, "No face detected", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
