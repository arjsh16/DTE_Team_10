import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog

def enhance_image(image):
    """Enhance image by removing noise and improving clarity."""
    # Convert to grayscale (optional, useful for processing)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Remove Gaussian Noise using Bilateral Filtering (preserves edges)
    enhanced = cv2.bilateralFilter(gray, 9, 75, 75)

    # Increase contrast using CLAHE (Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    contrast_enhanced = clahe.apply(enhanced)

    return contrast_enhanced

def remove_noise_image(image_path):
    """Reads an image, enhances it, and displays results."""
    # Load the image
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)

    # Process image
    enhanced_image = enhance_image(image)

    # Show results
    cv2.imshow("Original Image", image)
    cv2.imshow("Enhanced Image", enhanced_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def remove_noise_video(video_path):
    """Reads a video, applies noise removal & enhancement frame-by-frame, and displays the result."""
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Get FPS for proper playback speed
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # End of video

        # Enhance frame
        enhanced_frame = enhance_image(frame)

        # Display frames
        cv2.imshow("Original Video", frame)
        cv2.imshow("Enhanced Video", enhanced_frame)

        # Maintain original video speed
        if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def select_file():
    """Opens a file dialog to select an image or video from the device."""
    root = tk.Tk()
    root.withdraw()  # Hide the root window

    file_path = filedialog.askopenfilename(
        title="Select an image or video file", 
        filetypes=[
            ("Image & Video Files", "*.jpg;*.png;*.jpeg;*.bmp;*.mp4;*.avi;*.mov;*.mkv"),
            ("Image Files", "*.jpg;*.png;*.jpeg;*.bmp"),
            ("Video Files", "*.mp4;*.avi;*.mov;*.mkv")
        ]
    )

    if file_path:
        print(f"Selected file: {file_path}")
        if file_path.lower().endswith((".jpg", ".png", ".jpeg", ".bmp")):
            remove_noise_image(file_path)
        elif file_path.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
            remove_noise_video(file_path)
        else:
            print("Unsupported file format.")
    else:
        print("No file selected.")

# Run the file picker and process file
select_file()
