import os
import pygame
import logging
from gtts import gTTS
import numpy as np
import cv2


# Initialize pygame mixer
pygame.mixer.init()

# Define the folder path where audio files will be saved
AUDIO_FOLDER = "./audio_files"
os.makedirs(AUDIO_FOLDER, exist_ok=True)  # Ensure the folder exists, create if not

def play_audio(class_name):
    """
    Play audio corresponding to the detected class name.

    Parameters:
        class_name (str): The detected class name.

    Returns:
        None
    """
    try:
        # Replace underscores with spaces
        class_name = class_name.replace("_", " ")

        # Generate audio file name with the folder path
        audio_file = os.path.join(AUDIO_FOLDER, class_name + ".mp3")

        # Check if audio file exists
        if not os.path.isfile(audio_file):
            # If the audio file does not exist, generate it using gTTS
            tts_text = class_name + " ahead"  # Corrected a typo here: "a head" to "ahead"
            tts = gTTS(text=tts_text, lang='en')
            tts.save(audio_file)

        # Load and play the audio file
        pygame.mixer.music.load(audio_file)
        pygame.mixer.music.play()

        # Wait for audio to finish playing
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)  # Adjust tick rate as needed

        # Stop the mixer
        pygame.mixer.music.stop()

    except Exception as e:
        print(f"Error occurred during audio playback: {e}")
        
def display_sign_images(frame, detections, sign_region_height=100, sign_margin=10):
    """
    Display detected traffic sign images at the bottom of the frame.

    Args:
        frame: The input frame containing annotated traffic signs.
        detections: List of detections returned by the object detection model.
        sign_region_height: Height of the region to display sign images (default: 100).
        sign_margin: Margin between sign images (default: 10).

    Returns:
        Combined frame containing annotated frame and sign images region.
    """
    sign_images = []  # List to store detected sign images

    # Capture and store detected traffic sign images
    for detection in detections:
        x, y, w, h = map(int, detection[0])  # Convert coordinates to integers
        sign_image = frame[y:y+h, x:x+w]  # Crop the detected traffic sign image
        sign_images.append(sign_image)  # Add the cropped sign image to the list

    # Create an empty array for the sign region
    sign_region = np.zeros((sign_region_height, frame.shape[1], 3), dtype=np.uint8)

    # Display sign images in the sign region
    start_x = sign_margin
    for sign_image in sign_images:
        height, width, _ = sign_image.shape
        scale = sign_region_height / height
        resized_sign_image = cv2.resize(sign_image, (int(width * scale), sign_region_height))
        sign_region[:, start_x:start_x + resized_sign_image.shape[1]] = resized_sign_image
        start_x += resized_sign_image.shape[1] + sign_margin

    # Combine the frame with the sign region
    frame_with_sign_region = np.vstack([frame, sign_region])

    return frame_with_sign_region