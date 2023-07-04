# **Emotion detection**
This project uses a CNN model to classify emotions from pictures. The data fed to the model is taken from the webcam. Each second a picture is taken from the webcam, all faces from it are extracted and passed to the model to classify them. Then the predicted emotion is shown on the screen in the form of text.

---
## **Run**
To run the project, first download it 
```
git clone https://github.com/0Nikola0/emotion-detection.git
```
Then install the necessary dependencies
```
python -m pip install -r requirements.txt
```
Then run it with
```
python main.py
```

This should open a window showing you live footage from your webcam with text below each face with the model predictions.

---
## **Training**
If you want to re-train the model or make changes to it, you will need to do them in the `cnn_emotions_model.py` file and then run it with
```
python cnn_emotions_model.py
```
