# rock-paper-scissors-game
Rock-Paper-Scissors is a fundamental and enduring hand game played between two individuals, where each player simultaneously chooses one of the three options: Rock, Paper, or Scissors. The outcome is determined by the following principles:
- Rock prevails over Scissors
- Scissors triumph over Paper
- Paper prevails over Rock

## About the Project
This application is developed on Python version 3.10.

The purpose of this project is to develop an application that enables players to engage in a game of Rock-Paper-Scissors against the computer. The application utilizes the identification of hand keypoints to detect the hand signals of the player. 

The model used to detect hand keypoints are: `hand_keypoints_models.pt`. This model are trained from You Only Look Once Version 11 (YOLOv11) pretrained model (with COCO dataset). The training is passed through 50 epochs with image input size of (640, 640). At the completion of training, the model is able to achieve a pose detection precision of 0.748 and box precision of 0.914. 

## Getting Started
To set up the application on local machine, set up a virtual environment. For example, when using anaconda, virtual environment can be set up using the following command:

```
conda create -n "sample-env" python=3.10
```

When you're inside the virtual environment, execute the following command to install all the dependencies required:
```
pip install -r requirements.txt
```


## Usage
Run the application using the following command:

```
python main.py
```

To initiate the game, press the spacebar button. A countdown will commence, followed by a prompt to display your hand to the camera for a hand signal detection.

Upon presenting your hand, the game will determine the outcome of the match. A text display will indicate the winner or loser, which will remain visible for two seconds while the score is recorded. This ensures that any changes to hand signals after the result is displayed do not alter the recorded outcome. For instance, if you select rock and the game indicates a loss, it means the computer chose paper. In this scenario, the score will be updated accordingly. However, if you subsequently select scissors, despite the screen displaying a win, the previous result remains recorded, and the new score will not be updated.

To quit the game, press the "q" button.


## Features

Apart from the ability to detect hand signal, the application also performs score tracking until the game is terminated. This can be seen from the video as shown:

![Score Tracking](./img/save-score.gif)


## Technologies Used
The Tech Stack used are as follows:
- Python 3.10
- Ultralytics
- YOLOv11
- OpenCV-Python
- Google Colab (For Training the model)
