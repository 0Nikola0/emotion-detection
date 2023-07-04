import cv2
import numpy as np
from cnn_emotions_model import load_model


"""
Emotion Detection Using CNN Model and OpenCV

This script captures video from a webcam and performs real-time emotion detection on detected faces using a pre-trained Convolutional Neural Network (CNN) model. The emotions that can be detected are Angry, Disgust, Fear, Happy, Sad, Surprise, and Neutral.

Dependencies:
- Check the requirements.txt file

Usage:
1. Ensure that the pre-trained CNN model file 'cnn_emotions_model.py' is present in the same directory as this script.
2. Download the 'haarcascade_frontalface_default.xml' file from the OpenCV GitHub repository (https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml) and save it in the same directory as this script.
3. Run the script. The webcam will open, and the emotions of the detected faces will be displayed in real-time.

"""


def main() -> None:
    emotions = ("Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral")
    # Load the cascade
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    # Load the model
    model = load_model()

    # Capture video from webcam. 
    cap = cv2.VideoCapture(0)

    while True:
        # Read the frame
        _, frame = cap.read()
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect the faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        for (x, y, w, h) in faces:
            # Draw the rectangle around each face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

            # Extract face from frame
            face_img = gray[y:y+h, x:x+w]
            face_img = cv2.resize(face_img, (48, 48))
            face_img = np.expand_dims(face_img, axis = 0)
            
            # Predict emotion from the photo
            prediction = model.predict(face_img, verbose=0)
            emotion = emotions[np.argmax(prediction[0])]

            # Display emotion name
            cv2.putText(frame, emotion, (x, y+h+50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 240), 3)
            
        # Display frame
        cv2.imshow('Emotion detector', frame)
        
        # Stop if escape key is pressed
        k = cv2.waitKey(30) & 0xff
        if k==27:
            break

    # Release the VideoCapture object
    cap.release()

    # Destroy all windows open by OpenCV
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
