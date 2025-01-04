from ultralytics import YOLO
import cv2
import random

# List of colors with their RGB values
colors = [
    {"name": "Pastel Red", "rgb": (255, 179, 186)},
    {"name": "Pastel Green", "rgb": (186, 255, 201)},
    {"name": "Pastel Blue", "rgb": (179, 229, 255)},
    {"name": "Pastel Yellow", "rgb": (255, 253, 178)},
    {"name": "Pastel Cyan", "rgb": (178, 255, 255)},
    {"name": "Pastel Magenta", "rgb": (255, 178, 255)},
    {"name": "Bright White", "rgb": (255, 255, 255)},
    {"name": "Pastel Gray", "rgb": (211, 211, 211)},
    {"name": "Pastel Orange", "rgb": (255, 204, 153)},
    {"name": "Pastel Purple", "rgb": (204, 153, 255)},
    {"name": "Pastel Pink", "rgb": (255, 204, 229)},
    {"name": "Pastel Brown", "rgb": (222, 184, 135)},
    {"name": "Pastel Lime", "rgb": (204, 255, 153)},
    {"name": "Pastel Teal", "rgb": (153, 255, 204)},
    {"name": "Pastel Navy", "rgb": (153, 204, 255)},
    {"name": "Pastel Gold", "rgb": (255, 223, 186)},
    {"name": "Pastel Salmon", "rgb": (255, 182, 193)},
    {"name": "Pastel Beige", "rgb": (255, 245, 204)},
    {"name": "Pastel Olive", "rgb": (204, 255, 178)}
]

played = False

def get_color_by_name(name):
    for color in colors:
        if color["name"].lower() == name.lower():
            return color["rgb"]
    return (255, 255, 255)  # Default to white if not found

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

    # Check if there are keypoints detected
    if results[0].keypoints is not None and results[0].keypoints.data.numel() > 0:
        # Iterate through all detections (hands)
        
        for detection_index, keypoints in enumerate(results[0].keypoints.xy):  # Loop through each hand's keypoints
            
            for index, keypoint in enumerate(keypoints):
                x, y = int(keypoint[0]), int(keypoint[1])  # Convert to integers
                
                # Wrist
                if index == 0:
                    cv2.circle(frame, (x, y), radius=7, color=get_color_by_name("Pastel Magenta"), thickness=-1)

                # Thumb
                elif index < 5:  
                    cv2.circle(frame, (x, y), radius=7, color=get_color_by_name("Pastel Cyan"), thickness=-1)
            
                # Index
                if index>=5 and index < 9:
                    cv2.circle(frame, (x, y), radius=7, color=get_color_by_name("Pastel Yellow"), thickness=-1)

                # Middle
                elif index < 13:
                    cv2.circle(frame, (x, y), radius=7, color=get_color_by_name("Pastel Green"), thickness=-1)

                # Ring
                elif index < 17:
                    cv2.circle(frame, (x, y), radius=7, color=get_color_by_name("Pastel Blue"), thickness=-1)

                # Pinky
                else:
                    cv2.circle(frame, (x, y), radius=7, color=get_color_by_name("Pastel Navy"), thickness=-1)
            
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
            text_x = (frame_width - text_size[0]) // 2  # Center horizontally
            text_y = 20 + text_size[1]  # Slightly offset from the top

            user_choice = ""
            if (thumbs_is_straight and index_is_straight and middle_is_straight and ring_is_straight and pinky_is_straight):
                user_choice = "PAPER"
                print("PAPER")
                cv2.putText(frame, "PAPER", (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1.5, get_color_by_name("Pastel Red"), 5)

            elif (index_is_curled and middle_is_curled and ring_is_curled and pinky_is_curled):
                user_choice = "ROCK"
                print("ROCK")
                cv2.putText(frame, "ROCK", (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1.5, get_color_by_name("Pastel Red"), 5)
                
            elif (thumb_above_wrist):
                user_choice = "SCISSORS"
                print("SCISSORS")
                cv2.putText(frame, "SCISSORS", (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1.5, get_color_by_name("Pastel Red"), 5)
            else:
                user_choice = "ERROR" 
                print("Can't detect")
                cv2.putText(frame, "CAN'T DETECT", (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1.5, get_color_by_name("Pastel Red"), 5)

            

            if (user_choice == "PAPER" or user_choice == "SCISSORS" or user_choice == "ROCK"):
                
               
                if not played:
                    
                    random_int = random.randint(0, 2)
                    computer_string = "ROCK" if random_int == 0 else "PAPER" if random_int == 1 else "SCISSORS"

                    win_condition = (user_choice == "ROCK" and computer_string == "SCISSORS") or (user_choice == "SCISSORS" and computer_string == "PAPER") or (user_choice == "PAPER" and computer_string== "ROCK")
                    
                    played = True
                
                frame_height, frame_width, _ = frame.shape
                # Calculate text size and position
                

                if (user_choice == computer_string):
                    text_size = cv2.getTextSize("DRAW", cv2.FONT_HERSHEY_DUPLEX, 8, 15)[0]
                    text_width, text_height = text_size
                    text_x = (frame_width - text_width) // 2  # Center horizontally
                    text_y = (frame_height + text_height) // 2  # Center vertically (y considers text height)
                    cv2.putText(frame, "DRAW", (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX, 8, get_color_by_name("Pastel YELLOW"), 15)
                elif (win_condition):
                    text_size = cv2.getTextSize("YOU WIN", cv2.FONT_HERSHEY_DUPLEX, 8, 15)[0]
                    text_width, text_height = text_size
                    text_x = (frame_width - text_width) // 2  # Center horizontally
                    text_y = (frame_height + text_height) // 2  # Center vertically (y considers text height)
                    cv2.putText(frame, "YOU WIN", (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX, 8, get_color_by_name("Pastel GREEN"), 15)
                else:
                    text_size = cv2.getTextSize("YOU LOSE", cv2.FONT_HERSHEY_DUPLEX, 8, 15)[0]
                    text_width, text_height = text_size
                    text_x = (frame_width - text_width) // 2  # Center horizontally
                    text_y = (frame_height + text_height) // 2  # Center vertically (y considers text height)
                    cv2.putText(frame, "YOU LOSE", (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX, 8, get_color_by_name("Pastel Red"), 15)
            
                    # if user_choice == prev_random_string:
                    #     cv2.putText(frame, "DRAW", (text_x, text_y + 100), cv2.FONT_HERSHEY_SIMPLEX, 4, get_color_by_name("Pastel YELLOW"), 5)
                    # elif (win_condition):
                    #     cv2.putText(frame, "YOU WIN", (text_x, text_y + 100), cv2.FONT_HERSHEY_SIMPLEX, 4, get_color_by_name("Pastel GREEN"), 5)
                    # else:
                    #     cv2.putText(frame, "YOU LOSE", (text_x, text_y + 100), cv2.FONT_HERSHEY_SIMPLEX, 4, get_color_by_name("Pastel Red"), 5)



            
    else:
        print("No keypoints detected.")
        played = False


    # # Visualize the results on the frame
    # annotated_frame = results[0].plot()  # Annotated frame with predictions

    # Display the annotated frame
    cv2.imshow("YOLOv11 Real-Time Prediction", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
