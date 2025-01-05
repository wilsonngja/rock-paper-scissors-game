from ultralytics import YOLO
import cv2
import random
import time

# List of colors with their RGB values
colors = [
    {"name": "Pastel Red", "bgr": (186, 179, 255)},
    {"name": "Pastel Green", "bgr": (201, 255, 186)},
    {"name": "Pastel Blue", "bgr": (242, 235, 179)},
    {"name": "Pastel Yellow", "bgr": (140, 238, 255)},
    {"name": "Pastel Cyan", "bgr": (216, 216, 164)},
    {"name": "Pastel Magenta", "bgr": (194, 154, 244)},
    {"name": "Bright White", "bgr": (255, 255, 255)},
    {"name": "Pastel Gray", "bgr": (211, 211, 211)},
    {"name": "Pastel Orange", "bgr": (103, 192, 255)},
    {"name": "Pastel Purple", "bgr": (217, 156, 177)},
    {"name": "Pastel Pink", "bgr": (220, 209, 255)},
    {"name": "Pastel Brown", "bgr": (83, 105, 131)},
    {"name": "Pastel Lime", "bgr": (143, 236, 216)},
    {"name": "Pastel Teal", "bgr": (183, 183, 99)},
    {"name": "Pastel Navy", "bgr": (107, 66, 61)},
    {"name": "Pastel Gold", "bgr": (124, 210, 231)},
    {"name": "Pastel Salmon", "bgr": (178, 193, 246)},
    {"name": "Pastel Beige", "bgr": (200, 236, 254)},
    {"name": "Pastel Olive", "bgr": (130, 188, 188)}
]

played = False
gameStarted = False
gamePaused = False

countdown_start_time = None
countdown_value = None

game_paused_start_time = None
game_paused_value = None

ROCK = "ROCK"
SCISSORS = "SCISSORS"
PAPER = "PAPER"
START_GAME_TEXT = "Press '<space>' to start."
QUIT_GAME_TEXT = "Press 'q' to quit."
WIN = "YOU WIN"
LOSE = "YOU LOSE"
DRAW = "DRAW"

player_score = 0
computer_score = 0

def get_color_by_name(name):
    for color in colors:
        if color["name"].lower() == name.lower():
            return color["bgr"]
    return (255, 255, 255)  # Default to white if not found

def draw_text(text, scale, thickness, colour):
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, scale, thickness)[0]
    text_width, text_height = text_size
    text_x = (frame_width - text_width) // 2
    text_y = (frame_height + text_height) // 2
    cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX, scale, colour, thickness)

# Load a model
model = YOLO("./model/best.pt")  # load a pretrained model (recommended for training)

# # Train the model
# results = model.train(data="hand-keypoints.yaml", epochs=100, imgsz=320, batch=8)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, show=False)

    frame_height, frame_width, _ = frame.shape
    cv2.putText(frame, "YOUR SCORE", (30, 50), cv2.FONT_HERSHEY_TRIPLEX, 1, get_color_by_name("Pastel Green"), 2)
    cv2.putText(frame, str(player_score), (120, 100), cv2.FONT_HERSHEY_TRIPLEX, 1, get_color_by_name("Pastel Green"), 2)
    cv2.putText(frame, "COMPUTER SCORE", (frame_width - 330, 50), cv2.FONT_HERSHEY_TRIPLEX, 1, get_color_by_name("Pastel Red"), 2)
    cv2.putText(frame, str(computer_score), (frame_width - 190, 100), cv2.FONT_HERSHEY_TRIPLEX, 1, get_color_by_name("Pastel Red"), 2)


    # Countdown logic
    if countdown_start_time is not None:
        elapsed_time = time.time() - countdown_start_time
        countdown_value = max(3 - int(elapsed_time), 0)  # Countdown from 3 to 0
        if countdown_value == 0:
            gameStarted = True
            countdown_start_time = None  # Reset countdown
            played = False
            gamePaused = False
            countdown_value = None

    if not gameStarted:
        frame_height, frame_width, _ = frame.shape
        if countdown_value is not None:
            # Display the countdown
            draw_text(str(countdown_value), 5, 10, get_color_by_name("Pastel Gold"))
            
        else:
            # Display initial message
            text_size = cv2.getTextSize(QUIT_GAME_TEXT, cv2.FONT_HERSHEY_DUPLEX, 2, 7)[0]
            text_width, text_height = text_size
            text_x = (frame_width - text_width) // 2
            text_y = (frame_height + text_height) // 2
            cv2.putText(frame, QUIT_GAME_TEXT, (text_x, text_y - 50), cv2.FONT_HERSHEY_DUPLEX, 2, get_color_by_name("Pastel Pink"), 5)

            text_size = cv2.getTextSize(START_GAME_TEXT, cv2.FONT_HERSHEY_DUPLEX, 2, 5)[0]
            text_width, text_height = text_size
            text_x = (frame_width - text_width) // 2
            text_y = (frame_height + text_height) // 2
            cv2.putText(frame, START_GAME_TEXT, (text_x, text_y + 50), cv2.FONT_HERSHEY_DUPLEX, 2, get_color_by_name("Pastel Blue"), 5)
            
            
    else:
        boxes = results[0].boxes

        # Extract the tensor containing the xyxy coordinates
        coordinates_tensor = boxes.xyxy

        # Convert the tensor to a Python list for easier handling (if needed)
        coordinates_list = coordinates_tensor.tolist()
        if game_paused_start_time is not None:
            elapsed_time = time.time() - game_paused_start_time
            game_paused_value = max(2 - int(elapsed_time), 0)  # Countdown from 3 to 0
            if game_paused_value == 0:
                gameStarted = False
                game_paused_start_time = None  # Reset countdown


        # Check if there are keypoints detected
        if results[0].keypoints is not None and results[0].keypoints.data.numel() > 0:
            # Iterate through all detections (hands)
            for detection_index, keypoints in enumerate(results[0].keypoints.xy):  # Loop through each hand's keypoints
                
                for index, keypoint in enumerate(keypoints):
                    x, y = int(keypoint[0]), int(keypoint[1])  # Convert to integers
                    
                    # Wrist
                    if index == 0:
                        cv2.circle(frame, (x, y), radius=7, color=get_color_by_name("Pastel Beige"), thickness=-1)

                    # Thumb
                    elif index < 5:  
                        cv2.circle(frame, (x, y), radius=7, color=get_color_by_name("Pastel Lime"), thickness=-1)
                
                    # Index
                    if index < 9:
                        cv2.circle(frame, (x, y), radius=7, color=get_color_by_name("Pastel Yellow"), thickness=-1)

                    # Middle
                    elif index < 13:
                        cv2.circle(frame, (x, y), radius=7, color=get_color_by_name("Pastel Green"), thickness=-1)

                    # Ring
                    elif index < 17:
                        cv2.circle(frame, (x, y), radius=7, color=get_color_by_name("Pastel Blue"), thickness=-1)

                    # Pinky
                    else:
                        cv2.circle(frame, (x, y), radius=7, color=get_color_by_name("Pastel Salmon"), thickness=-1)
                
                # To detect Paper
                thumbs_is_straight = keypoints[1][0] > keypoints[2][0] > keypoints[3][0] > keypoints[4][0] or keypoints[1][0] < keypoints[2][0] < keypoints[3][0] < keypoints[4][0]
                index_is_straight = keypoints[5][1] > keypoints[6][1] > keypoints[7][1] > keypoints[8][1]
                middle_is_straight = keypoints[9][1] > keypoints[10][1] > keypoints[11][1] > keypoints[12][1]
                ring_is_straight = keypoints[13][1] > keypoints[14][1] > keypoints[15][1] > keypoints[16][1]
                pinky_is_straight = keypoints[17][1] > keypoints[18][1] > keypoints[19][1] > keypoints[20][1]
                
                # To detect Rock
                index_is_curled = keypoints[8][1] > keypoints[5][1]
                middle_is_curled = keypoints[12][1] > keypoints[9][1]
                ring_is_curled = keypoints[16][1] > keypoints[10][1]
                pinky_is_curled = keypoints[20][1] > keypoints[17][1]

                # To detect Scissors
                thumb_above_wrist = keypoints[0][1] > keypoints[4][1] and abs(keypoints[0][0] - keypoints[4][0]) < 100
        
                frame_height, frame_width, _ = frame.shape
                # Calculate text size and position
                text_size = cv2.getTextSize("DETECTED", cv2.FONT_HERSHEY_SIMPLEX, 1, 5)[0]
                text_x = int((coordinates_list[0][0] + coordinates_list[0][2]) // 2)
                text_y = int(coordinates_list[0][1] - 10)
                
                user_choice = ""
                if (thumbs_is_straight and index_is_straight and middle_is_straight and ring_is_straight and pinky_is_straight):
                    user_choice = "PAPER"
                    cv2.putText(frame, "PAPER", (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1.5, get_color_by_name("Pastel Cyan"), 5)

                elif (index_is_curled and middle_is_curled and ring_is_curled and pinky_is_curled):
                    user_choice = "ROCK"
                    cv2.putText(frame, "ROCK", (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1.5, get_color_by_name("Pastel Cyan"), 5)
                    
                elif (thumb_above_wrist):
                    user_choice = "SCISSORS"
                    cv2.putText(frame, "SCISSORS", (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1.5, get_color_by_name("Pastel Cyan"), 5)
                else:
                    user_choice = "ERROR" 
                    cv2.putText(frame, "CAN'T DETECT", (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1.5, get_color_by_name("Pastel Red"), 5)

                

                if (user_choice == "PAPER" or user_choice == "SCISSORS" or user_choice == "ROCK"):
                    
                
                    if not played:
                        
                        random_int = random.randint(0, 2)
                        computer_string = "ROCK" if random_int == 0 else "PAPER" if random_int == 1 else "SCISSORS"

                        win_condition = (user_choice == "ROCK" and computer_string == "SCISSORS") or (user_choice == "SCISSORS" and computer_string == "PAPER") or (user_choice == "PAPER" and computer_string== "ROCK")
                        if (user_choice == computer_string):
                            draw_text(DRAW, 8, 15, get_color_by_name("Pastel Yellow"))

                        elif (win_condition):
                            draw_text(WIN, 8, 15, get_color_by_name("Pastel Green"))
                            player_score += 1

                        else:
                            draw_text(LOSE, 8, 15, get_color_by_name("Pastel Red"))
                            computer_score += 1

                        played = True
                        
                    elif not gamePaused:
                        game_paused_start_time = time.time()
                        gamePaused = True
                              
                    if (user_choice == computer_string):
                        draw_text(DRAW, 8, 15, get_color_by_name("Pastel Yellow"))

                    elif (win_condition):
                        draw_text(WIN, 8, 15, get_color_by_name("Pastel Green"))

                    else:
                        draw_text(LOSE, 8, 15, get_color_by_name("Pastel Red"))

   
    # Display the annotated frame
    cv2.imshow("YOLOv11 Real-Time Prediction", frame)
    key = cv2.waitKey(1) & 0xFF

    # Exit on pressing 'q'
    if key == ord("q"):
        break

    # Start countdown when space is pressed
    if key == ord(" ") and countdown_start_time is None and not gameStarted:
        countdown_start_time = time.time()

# Release resources
cap.release()
cv2.destroyAllWindows()
