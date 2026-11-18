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
                (ast.literal_eval(row['left_shoulder']), ast.literal_eval(row['right_shoulder'])),
                (ast.literal_eval(row['left_shoulder']), ast.literal_eval(row['left_elbow'])), (ast.literal_eval(row['left_elbow']), ast.literal_eval(row['left_wrist'])),
                (ast.literal_eval(row['right_shoulder']), ast.literal_eval(row['right_elbow'])), (ast.literal_eval(row['right_elbow']), ast.literal_eval(row['right_wrist']))            
            ]

        for pt1, pt2 in Body_drawn:
            x1, y1 = int(pt1[0] * width), int(pt1[1] * height)
            x2, y2 = int(pt2[0] * width), int(pt2[1] * height)
            cv2.line(frame, (x1, y1), (x2, y2), (255, 255, 191), 1)

        cv2.imshow("drawing", frame)

        while True:
            key = cv2.waitKey(0)
            if cv2.waitKey(0) == ord('q'):
                break

cv2.destroyAllWindows()




    
         