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


To test if the application is running successfully, execute the `main.py` using:
```
python main.py
```

## Usage

## Features

## Technologies Used

