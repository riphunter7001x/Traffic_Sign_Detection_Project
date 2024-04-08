import cv2
import supervision as sv
import os
import pygame
from src.utils import play_audio
from src.utils import display_sign_images
from src.model import load_model
import logging

# Constants
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

# Configure logging
LOGS_FOLDER = r".\logs"
LOG_FILE = os.path.join(LOGS_FOLDER, "traffic_sign_detection.log")
os.makedirs(LOGS_FOLDER, exist_ok=True)

logging.basicConfig(filename=LOG_FILE, level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    """
    Perform real-time object detection using a webcam feed and play audio cues for detected objects.

    This function captures video frames from a webcam, performs object detection using a YOLO model,
    annotates the frames with bounding boxes and labels of detected objects, and plays audio cues for
    newly detected objects. The audio cues are played based on the class names of the detected objects.
    Only one audio cue is played for each unique detection, and repeated detections of the same object
    do not trigger additional audio playback.

    Returns:
        None
    """
    logging.info("Starting traffic sign detection application...")

    # Initialize cap here
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    # Load YOLO model
    try:
        model = load_model()
    except Exception as e:
        logging.error(f"Failed to load YOLO model: {str(e)}")
        return

    # Initialize BoxAnnotator
    box_annotator = sv.BoxAnnotator(
        thickness=2,
        text_thickness=2,
        text_scale=1
    )

    last_played_audio = None  # Variable to store the last played audio
    # Main Loop
    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # Resize frame for faster processing
        resized_frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

        # Perform object detection
        result = model(resized_frame, agnostic_nms=True)[0]
        detections = sv.Detections.from_yolov8(result)

        # Initialize variables to store modified labels and detected classes
        modified_labels = []
        detected_classes = set()

        # Loop through detections to modify labels and store detected classes
        for _, confidence, class_id, _ in detections:
            class_name = model.model.names[class_id]
            if class_name == "LOAD_LIMIT":
                modified_labels.append("speed_limit" + f" {confidence:.2f}")
                detected_classes.add("speed_limit")
            else:
                modified_labels.append(f"{class_name} {confidence:.2f}")
                detected_classes.add(class_name)

        # Annotate frame
        annotated_frame = box_annotator.annotate(
            scene=frame,
            detections=detections,
            labels=modified_labels
        )

        # Display annotated frame
        frame_with_sign_region = display_sign_images(annotated_frame, detections)
        cv2.imshow("Traffic Sign Detection", frame_with_sign_region)

        # Play audio for new detections
        if detected_classes and detected_classes != last_played_audio:
            class_name = next(iter(detected_classes))
            play_audio(class_name)
            last_played_audio = detected_classes
            logging.info(f"Audio played for detected class: {class_name}")
            
        key = cv2.waitKey(10)  # Adjust delay as needed

        # Exit loop on 'q' key press
        if key & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

    logging.info("Traffic sign detection application finished.")


if __name__ == "__main__":
    main()
