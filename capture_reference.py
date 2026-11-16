import math
import cv2
import numpy as np
from time import time
import mediapipe as mp
import matplotlib.pyplot as pit

BaseOptions = mp.tasks.BaseOptions
HolisticLandmarker = mp.tasks.vision.HolisticLandmarker
HolisticLandmarkerOptions = mp.tasks.vision.HolisticLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HolisticLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="holistic_landmarker.task"),
    running_mode=VisionRunningMode.VIDEO,
    min_face_detection_confidence=0.5,
    min_pose_detection_confidence=0.5,
)

POSE_CONNECTIONS_BODY = [
    # Shoulders and arms
    (11, 12), (11, 13), (13, 15), # Left arm and hands
    (12, 14), (14, 16),          # Right arm and hands
    # Torso
    (11, 23), (12, 24), (23, 24),
    # Legs
    (23, 25), (25, 27),            # Left leg
    (24, 26), (26, 28),            # Right leg
]

POSE_CONNECTIONS_FACE = [
    (70, 63), (63, 105), (105, 66), (66, 107), #Left Eyebrow
    (46, 53), (53, 52), (52, 65), (65, 55), #Left eyebrow bottom

    (336, 296), (296, 334), (334, 293), (293, 300), #Right eyebrow
    (285, 295), (295, 282), (282, 283), (283, 276), #Right eyebrow bottom

    (362, 382), (382, 381), (381, 380), (380, 374), (374, 373), (373, 390), (390, 249), (249, 263), (263, 466),
    (466, 388), (388, 387), (387,  386), (386, 385), (385, 384), (384, 398), (398, 362), #Right eye

    (133, 173), (173, 157), (157, 158), (158, 159), (159, 160), (160, 161), (161, 246), (246, 33),
    (33, 7), (7, 163), (163, 144), (144, 145), (145, 153), (153, 154), (154, 155), #Left eye

    (310, 311), (311, 312), (312, 13), (13, 82), (82, 81), (81, 80), (80, 191), (191, 78), (78, 95), (95, 88),
    (88, 178), (178, 87), (87, 14), (14, 317), (317, 402), (402, 318), (318, 324), (324, 308), (308, 415), (415,  310), #Inner lip
]

POSE_CONNECTIONS_HAND = [
    (0, 1), (1, 2), (2, 3), (3, 4), #thumb
    (0, 5), (5, 6), (6, 7), (7, 8), # pointer
    (5, 9), (9, 10), (10, 11), (11, 12), #middle
    (9, 13), (13, 14), (14, 15), (15, 16), #ring
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17) #pinky

]


cap = cv2.VideoCapture(0)

with HolisticLandmarker.create_from_options(options) as landmarker:
    frame_timestamp_ms = 0

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            print("Couldnt capture camera")
            break

        frame = cv2.flip(frame, 1)

        height, width, _ = frame.shape

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        frame_timestamp_ms += 33

        result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

        if result.face_landmarks:
            face = result.face_landmarks

            for start_idx, end_idx in POSE_CONNECTIONS_FACE: #so it walks through the pair in the pose_connects array
                pt1 = face[start_idx] 
                pt2 = face[end_idx]
                x1, y1 = int(pt1.x * width), int(pt1.y * height)
                x2, y2 = int(pt2.x * width), int(pt2.y * height)
                cv2.line(frame, (x1, y1), (x2, y2), (255, 255, 191), 1)  



        if result.pose_landmarks:
            body = result.pose_landmarks

            for start_idx, end_idx in POSE_CONNECTIONS_BODY:
                pt1 = body[start_idx]
                pt2 = body[end_idx]
                x1, y1 = int(pt1.x * width), int(pt1.y * height)
                x2, y2 = int(pt2.x * width), int(pt2.y * height)
                cv2.line(frame, (x1, y1), (x2, y2), (0, 183, 235), 2)

            

        if result.left_hand_landmarks:
                lefthand = result.left_hand_landmarks
                
                for start_idx, end_idx in POSE_CONNECTIONS_HAND:
                    pt1 = lefthand[start_idx]
                    pt2 = lefthand[end_idx]
                    x1, y1 = int(pt1.x * width), int(pt1.y * height)
                    x2, y2 = int(pt2.x * width), int(pt2.y * height)
                    cv2.line(frame, (x1, y1), (x2, y2), (255, 255, 191), 2)
                
                

        if result.right_hand_landmarks:
                righthand = result.right_hand_landmarks
                        
                for start_idx, end_idx in POSE_CONNECTIONS_HAND:
                    pt1 = righthand[start_idx]
                    pt2 = righthand[end_idx]
                    x1, y1 = int(pt1.x * width), int(pt1.y * height)
                    x2, y2 = int(pt2.x * width), int(pt2.y * height)
                    cv2.line(frame, (x1, y1), (x2, y2), (255, 255, 191), 2)
                        
                


        cv2.imshow("webcam", frame)
        if cv2.waitKey(1) == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()




