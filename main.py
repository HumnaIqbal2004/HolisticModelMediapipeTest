import math
import cv2
import numpy as np
from time import time
import mediapipe as mp
import matplotlib.pyplot as pit

def font():
    font = cv2.FONT_HERSHEY_COMPLEX
    cv2.putText(frame, f"frame: {current_frame}", (50, 50), font, 1, (0, 255, 0), 2, lineType=cv2.LINE_AA)


BaseOptions = mp.tasks.BaseOptions
HolisticLandmarker = mp.tasks.vision.HolisticLandmarker
HolisticLandmarkerOptions = mp.tasks.vision.HolisticLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HolisticLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="holistic_landmarker.task"),
    running_mode=VisionRunningMode.VIDEO,
    min_face_detection_confidence=0.8,
    min_pose_detection_confidence=0.8,
    min_pose_presence_confidence=0.5,
    min_pose_suppression_threshold=0.7
    
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

cap = cv2.VideoCapture('assets/dance3.mp4')


with HolisticLandmarker.create_from_options(options) as landmarker:
    frame_timestamp_ms = 0
    paused = False
    
    
    
    while cap.isOpened():
            ret, frame = cap.read()            

            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            else:
                fps= cap.get(cv2.CAP_PROP_FPS)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                

            frame = cv2.flip(frame, 1)
            current_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)

            font()
        
            height, width, _ = frame.shape #grabs height and and width of video frame
        
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
            frame_timestamp_ms += int(1000 / fps)
        
            result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)
            
        
            if result.face_landmarks:
                face = result.face_landmarks

                inner_corner_right = ((face[78].x), (face[78]))
                inner_corner_left = ((face[308].x), (face[308]))
                center_top = ((face[13].x), (face[13]))
                center_bottom = ((face[14].x), (face[14]))

                for start_idx, end_idx in POSE_CONNECTIONS_FACE: #so it walks through the pair in the pose_connects array
                    pt1 = face[start_idx] 
                    pt2 = face[end_idx]
                    x1, y1 = int(pt1.x * width), int(pt1.y * height)
                    x2, y2 = int(pt2.x * width), int(pt2.y * height)
                    cv2.line(frame, (x1, y1), (x2, y2), (255, 255, 191), 1)  
                print(f"Current Frame: {current_frame} | Inner_corner_R: {inner_corner_right} | Inner_corner_L: {inner_corner_left} | Center_top: {center_top} | Center_bottom: {center_bottom}")
        
        
            if result.pose_landmarks:
                body = result.pose_landmarks

                left_shoulder = ((body[11].x), (body[11].y))
                right_shoulder = ((body[12].x), (body[12].y))
                left_elbow = ((body[13].x), (body[13].y))
                right_elbow = ((body[14].x), (body[14].y))
                left_wrist = ((body[15].x), (body[15].y))
                right_wrist = ((body[16].x), (body[16].y))

                for start_idx, end_idx in POSE_CONNECTIONS_BODY:
                    pt1 = body[start_idx]
                    pt2 = body[end_idx]
                    x1, y1 = int(pt1.x * width), int(pt1.y * height)
                    x2, y2 = int(pt2.x * width), int(pt2.y * height)
                    cv2.line(frame, (x1, y1), (x2, y2), (0, 183, 235), 2)
                print(f"Current Frame: {current_frame} | Left_Elbow: {left_elbow} | Left_Wrist: {left_wrist} | Shoulders: {left_shoulder}, {right_shoulder} | Right_Elbow: {right_elbow} | Right_Wrist: {right_wrist}")

                
                    
        
            if result.left_hand_landmarks:
                lefthand = result.left_hand_landmarks

                hand_wrist_L = (lefthand[0].x), (lefthand[0].y)
                left_thumb = (lefthand[4].x), (lefthand[4].y)
                left_poiner = (lefthand[8].x), (lefthand[8].y)
                left_middle = (lefthand[12].x), (lefthand[12].y)
                left_ring = (lefthand[16].x), (lefthand[16].y)
                left_pinky = (lefthand[20].x), (lefthand[20].y)


                for start_idx, end_idx in POSE_CONNECTIONS_HAND:
                    pt1 = lefthand[start_idx]
                    pt2 = lefthand[end_idx]
                    x1, y1 = int(pt1.x * width), int(pt1.y * height)
                    x2, y2 = int(pt2.x * width), int(pt2.y * height)
                    cv2.line(frame, (x1, y1), (x2, y2), (255, 255, 191), 2)
                print(f"Current Frame: {current_frame} | Hand_wrist: {hand_wrist_L} | Left_thumb: {left_thumb} | Left_pointer: {left_poiner} | Left_middle: {left_middle} | Left_ring: {left_ring} | Left_pinky: {left_pinky}")  
                        
        
            if result.right_hand_landmarks:
                righthand = result.right_hand_landmarks

                hand_wrist_R = (righthand[0].x), (righthand[0].y)
                right_thumb = (righthand[4].x), (righthand[4].y)
                right_poiner = (righthand[8].x), (righthand[8].y)
                right_middle = (righthand[12].x), (righthand[12].y)
                right_ring = (righthand[16].x), (righthand[16].y)
                right_pinky = (righthand[20].x), (righthand[20].y)
                                
                for start_idx, end_idx in POSE_CONNECTIONS_HAND:
                    pt1 = righthand[start_idx]
                    pt2 = righthand[end_idx]
                    x1, y1 = int(pt1.x * width), int(pt1.y * height)
                    x2, y2 = int(pt2.x * width), int(pt2.y * height)
                    cv2.line(frame, (x1, y1), (x2, y2), (255, 255, 191), 2)
                print(f"Current Frame: {current_frame} | Hand_wrist_R: {hand_wrist_R} | Right_thumb: {right_thumb} | Right_pointer: {right_poiner} | Right_middle: {right_middle} | Right_ring: {right_ring} | Right_pinky: {right_pinky}")

            print("---------------")

            
            delay = 0 if paused else int(1000/ fps)
            key = cv2.waitKey(delay)

            cv2.imshow("video", frame)
            if key == ord('q'):
                break

            elif key == ord("p"):
                paused = not paused

            elif key == ord("f"):
                paused = True


            
                
                
                


            
                
                
                

print(f"Frame rate: {fps}")
print(f"Total Frames: {total_frames}")
cap.release()
cv2.destroyAllWindows()




