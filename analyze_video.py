import cv2
from deepface import DeepFace
import numpy as np

# Initialize the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Cannot open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    try:
        # We are passing enforce_detection=False so that the program doesn't crash if no face is found.
        # It will simply return the original frame.
        analysis = DeepFace.analyze(
            img_path = frame,
            actions = ['emotion'],
            enforce_detection=False
        )

        # The result is a list of dictionaries, one for each face detected.
        # We'll focus on the first face found.
        if isinstance(analysis, list) and analysis:
            first_face = analysis[0]
            dominant_emotion = first_face['dominant_emotion']
            # Get the bounding box of the face
            face_region = first_face['region']
            x, y, w, h = face_region['x'], face_region['y'], face_region['w'], face_region['h']

            # Draw a rectangle around the face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # Prepare the text to display
            text = f"Emotion: {dominant_emotion}"

            # Put the text on the frame
            cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    except Exception as e:
        # Sometimes deepface throws an error if the face is not clear.
        # We'll just print it and continue.
        print(f"Error during analysis: {e}")


    # Display the resulting frame
    cv2.imshow('AI Interviewer - Facial Analysis', frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()