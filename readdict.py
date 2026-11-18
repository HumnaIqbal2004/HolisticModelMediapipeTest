import csv
import numpy as np
import mediapipe as mp
import cv2
import ast

blank = cv2.imread('assets/Solid_black.png')
height, width, _ = blank.shape

with open("video_pose_data.csv", mode="r", newline="") as file:
    reader = csv.DictReader(file)

        
    for row in reader:

        frame = blank.copy()
        Body_drawn = [
                (ast.literal_eval(row['left_shoulder']), ast.literal_eval(row['right_shoulder'])), #Shoulder
                (ast.literal_eval(row['left_shoulder']), ast.literal_eval(row['left_elbow'])), (ast.literal_eval(row['left_elbow']), ast.literal_eval(row['left_wrist'])), #left arm
                (ast.literal_eval(row['right_shoulder']), ast.literal_eval(row['right_elbow'])), (ast.literal_eval(row['right_elbow']), ast.literal_eval(row['right_wrist'])), #right arm

                (ast.literal_eval(row['inner_corner_right']), ast.literal_eval(row['center_top'])), (ast.literal_eval(row['inner_corner_right']), ast.literal_eval(row['center_bottom'])), (ast.literal_eval(row['inner_corner_left']), ast.literal_eval(row['center_top'])), (ast.literal_eval(row['inner_corner_left']), ast.literal_eval(row['center_bottom'])), #mouth

                (ast.literal_eval(row['hand_wrist_L']), ast.literal_eval(row['left_thumb_knuckle'])), (ast.literal_eval(row['hand_wrist_L']), ast.literal_eval(row['left_pointer_knuckle'])), (ast.literal_eval(row['hand_wrist_L']), ast.literal_eval(row['left_pinky_knuckle'])), (ast.literal_eval(row['left_pointer_knuckle']), ast.literal_eval(row['left_pinky_knuckle'])), #Left Hand
                (ast.literal_eval(row['left_thumb_knuckle']), ast.literal_eval(row['left_thumb_joint'])), (ast.literal_eval(row['left_thumb_joint']), ast.literal_eval(row['left_thumb'])), #Left Thumb
                (ast.literal_eval(row['left_pointer_knuckle']), ast.literal_eval(row['left_pointer_joint'])), (ast.literal_eval(row['left_pointer_joint']), ast.literal_eval(row['left_pointer'])), #Left Pointer
                (ast.literal_eval(row['left_middle_knuckle']), ast.literal_eval(row['left_middle_joint'])), (ast.literal_eval(row['left_middle_joint']), ast.literal_eval(row['left_middle'])), #left middle
                (ast.literal_eval(row['left_ring_knuckle']), ast.literal_eval(row['left_ring_joint'])), (ast.literal_eval(row['left_ring_joint']), ast.literal_eval(row['left_ring'])), #left ring
                (ast.literal_eval(row['left_pinky_knuckle']), ast.literal_eval(row['left_pinky_joint'])), (ast.literal_eval(row['left_pinky_joint']), ast.literal_eval(row['left_pinky'])), #left pinky

                (ast.literal_eval(row['hand_wrist_R']), ast.literal_eval(row['right_thumb_knuckle'])), (ast.literal_eval(row['hand_wrist_R']), ast.literal_eval(row['right_pointer_knuckle'])), (ast.literal_eval(row['hand_wrist_R']), ast.literal_eval(row['right_pinky_knuckle'])), (ast.literal_eval(row['right_pointer_knuckle']), ast.literal_eval(row['right_pinky_knuckle'])), #Right Hand
                (ast.literal_eval(row['right_thumb_knuckle']), ast.literal_eval(row['right_thumb_joint'])), (ast.literal_eval(row['right_thumb_joint']), ast.literal_eval(row['right_thumb'])), #right Thumb
                (ast.literal_eval(row['right_pointer_knuckle']), ast.literal_eval(row['right_pointer_joint'])), (ast.literal_eval(row['right_pointer_joint']), ast.literal_eval(row['right_pointer'])), #right Pointer
                (ast.literal_eval(row['right_middle_knuckle']), ast.literal_eval(row['right_middle_joint'])), (ast.literal_eval(row['right_middle_joint']), ast.literal_eval(row['right_middle'])), #right middle
                (ast.literal_eval(row['right_ring_knuckle']), ast.literal_eval(row['right_ring_joint'])), (ast.literal_eval(row['right_ring_joint']), ast.literal_eval(row['right_ring'])), #right ring
                (ast.literal_eval(row['right_pinky_knuckle']), ast.literal_eval(row['right_pinky_joint'])), (ast.literal_eval(row['right_pinky_joint']), ast.literal_eval(row['right_pinky'])), #right pinky

                
            ]

        for pt1, pt2 in Body_drawn:
            x1, y1 = int(pt1[0] * width), int(pt1[1] * height)
            x2, y2 = int(pt2[0] * width), int(pt2[1] * height)
            cv2.line(frame, (x1, y1), (x2, y2), (255, 255, 191), 1)

        cv2.imshow("drawing", frame)

        while True:
            key = cv2.waitKey(0)
            if key == ord('q') or key == ord('d'):
                break

        if key == ord('q'):
            break

cv2.destroyAllWindows()




    
         