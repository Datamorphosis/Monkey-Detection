from flask import Flask, Response
import cv2
from ultralytics import YOLO
import logging
import pygame

app = Flask(__name__)

# Configure the logging
log_file = 'detection.log'
logging.basicConfig(filename=log_file, level=logging.INFO,
                    format='%(asctime)s - %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

# Disable Flask server logs and HTTP request logs
werkzeug_logger = logging.getLogger('werkzeug')
werkzeug_logger.setLevel(logging.ERROR)

# Initialize Pygame mixer for playing sound
pygame.mixer.init()

def generate_frames():
    video_path = './monkey_in.mp4'
    model_path = './monkey_best.pt'

    cap = cv2.VideoCapture(video_path)
    model = YOLO(model_path)
    threshold = 0.2

    frame_count = 1  # Start frame count from 1

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)[0]

        monkeys_in_frame = 0

        for result in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = result
            if score > threshold:
                monkeys_in_frame += 1
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
                cv2.putText(frame, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

        # Log the count of monkeys detected in the frame along with timestamp
        log_message = f'Frame {frame_count}: {monkeys_in_frame} Monkey(s) detected'
        logging.info(log_message)

        if monkeys_in_frame > 0:
            # Play sound when monkeys are detected using Pygame mixer
            pygame.mixer.Sound('./siren.mp3').play()

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        frame_count += 1  # Increment frame count

    cap.release()

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
