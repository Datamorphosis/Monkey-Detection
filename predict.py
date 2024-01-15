import os
import cv2
from ultralytics import YOLO
import logging
import json
from playsound import playsound

def detect_and_annotate(video_path, output_path, model_path):
    # Create a logger
    logger = logging.getLogger('MonkeyDetection')
    logger.setLevel(logging.INFO)

    # Create a file handler for the log file
    file_handler = logging.FileHandler('monkey_detection.log')
    file_handler.setLevel(logging.INFO)

    # Create a formatter and set it for the handler
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(formatter)

    # Add the file handler to the logger
    logger.addHandler(file_handler)

    # Create an empty list to store log messages
    log_data = []

    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    H, W, _ = frame.shape
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'avc1'), int(cap.get(cv2.CAP_PROP_FPS)), (W, H))

    model = YOLO(model_path)
    threshold = 0.2

    #sound="./Siren Sound.mp3"

    frame_count = 0

    while ret:
        frame_count += 1
        results = model(frame)[0]

        monkeys_in_frame = 0

        for result in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = result

            if score > threshold:
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
                cv2.putText(frame, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
                #playsound(sound)

                monkeys_in_frame += 1

        if monkeys_in_frame > 0:
            message = f'Frame {frame_count}: {monkeys_in_frame} Monkey(s) detected'
            logger.info(message)
            log_data.append({'frame': frame_count, 'monkeys_detected': monkeys_in_frame, 'message': message})

        out.write(frame)
        # Show the frame with detections in a window
        cv2.imshow('Detected Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit window
            break

        ret, frame = cap.read()



    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # Save log data as a JSON file
    with open('monkey_detection.json', 'w') as json_file:
        json.dump(log_data, json_file, indent=4)


def main():
    video_path = './monkey_in.mp4'
    output_path = './monkey_out.mp4'
    model_path = './monkey_best.pt'

    detect_and_annotate(video_path, output_path, model_path)


if __name__ == "__main__":
    main()
