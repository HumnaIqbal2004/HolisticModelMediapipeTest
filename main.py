import math
import cv2
import numpy as np
from time import time
import mediapipe as mp
import matplotlib.pyplot as pit
import csv

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

cap = cv2.VideoCapture('assets/dance.mp4')


with HolisticLandmarker.create_from_options(options) as landmarker:
    frame_timestamp_ms = 0
    paused = False
    recorded_data = []
    
    
    
    while cap.isOpened():
            ret, frame = cap.read()            

            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            else:
                fps= cap.get(cv2.CAP_PROP_FPS)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                

            frame = cv2.flip(frame, 1)
            current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

            font()
        
            height, width, _ = frame.shape #grabs height and and width of video frame
        
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
            frame_timestamp_ms += int(1000 / fps)
        
            result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

            frame_data = {"Current Frame": current_frame}
        
            if result.face_landmarks:
                face = result.face_landmarks
                

                frame_data["inner_corner_right"] = ((face[78].x), (face[78].y))
                frame_data["inner_corner_left"] = ((face[308].x), (face[308].y))
                frame_data["center_top"] = ((face[13].x), (face[13].y))
                frame_data["center_bottom"] = ((face[14].x), (face[14].y))

                for start_idx, end_idx in POSE_CONNECTIONS_FACE: #so it walks through the pair in the pose_connects array
                    pt1 = face[start_idx] 
                    pt2 = face[end_idx]
                    x1, y1 = int(pt1.x * width), int(pt1.y * height)
                    x2, y2 = int(pt2.x * width), int(pt2.y * height)
                    cv2.line(frame, (x1, y1), (x2, y2), (255, 255, 191), 1)  
                
        
            if result.pose_landmarks:
                body = result.pose_landmarks

                frame_data["left_shoulder"] = ((body[11].x), (body[11].y))
                frame_data["right_shoulder"] = ((body[12].x), (body[12].y))
                frame_data["left_elbow"] = ((body[13].x), (body[13].y))
                frame_data["right_elbow"] = ((body[14].x), (body[14].y))
                frame_data["left_wrist"] = ((body[15].x), (body[15].y))
                frame_data["right_wrist"] = ((body[16].x), (body[16].y))

                for start_idx, end_idx in POSE_CONNECTIONS_BODY:
                    pt1 = body[start_idx]
                    pt2 = body[end_idx]
                    x1, y1 = int(pt1.x * width), int(pt1.y * height)
                    x2, y2 = int(pt2.x * width), int(pt2.y * height)
                    cv2.line(frame, (x1, y1), (x2, y2), (0, 183, 235), 2)
                
                
                    
        
            if result.left_hand_landmarks:
                lefthand = result.left_hand_landmarks

                frame_data["hand_wrist_L"] = (lefthand[0].x), (lefthand[0].y)
                frame_data["left_thumb"] = (lefthand[4].x), (lefthand[4].y)
                frame_data["left_thumb_knuckle"] = (lefthand[2].x), (lefthand[2].y)
                frame_data["left_thumb_joint"] = (lefthand[3].x), (lefthand[3].y)

                frame_data["left_pointer"] = (lefthand[8].x), (lefthand[8].y)
                frame_data["left_pointer_knuckle"] = (lefthand[5].x), (lefthand[5].y)
                frame_data["left_pointer_joint"] = (lefthand[6].x), (lefthand[6].y)

                frame_data["left_middle"] = (lefthand[12].x), (lefthand[12].y)
                frame_data["left_middle_knuckle"] = (lefthand[9].x), (lefthand[9].y)
                frame_data["left_middle_joint"] = (lefthand[10].x), (lefthand[10].y)

                frame_data["left_ring"] = (lefthand[16].x), (lefthand[16].y)
                frame_data["left_ring_knuckle"] = (lefthand[13].x), (lefthand[13].y)
                frame_data["left_ring_joint"] = (lefthand[14].x), (lefthand[14].y)

                frame_data["left_pinky"] = (lefthand[20].x), (lefthand[20].y)
                frame_data["left_pinky_knuckle"] = (lefthand[17].x), (lefthand[17].y)
                frame_data["left_pinky_joint"] = (lefthand[18].x), (lefthand[18].y)


                for start_idx, end_idx in POSE_CONNECTIONS_HAND:
                    pt1 = lefthand[start_idx]
                    pt2 = lefthand[end_idx]
                    x1, y1 = int(pt1.x * width), int(pt1.y * height)
                    x2, y2 = int(pt2.x * width), int(pt2.y * height)
                    cv2.line(frame, (x1, y1), (x2, y2), (255, 255, 191), 2)
                
        
            if result.right_hand_landmarks:
                righthand = result.right_hand_landmarks

                frame_data["hand_wrist_R"] = (righthand[0].x), (righthand[0].y)
                frame_data["right_thumb"] = (righthand[4].x), (righthand[4].y)
                frame_data["right_thumb_knuckle"] = (righthand[2].x), (righthand[2].y)
                frame_data["right_thumb_joint"] = (righthand[3].x), (righthand[3].y)


                frame_data["right_pointer"] = (righthand[8].x), (righthand[8].y)
                frame_data["right_pointer_knuckle"] = (righthand[5].x), (righthand[5].y)
                frame_data["right_pointer_joint"] = (righthand[6].x), (righthand[6].y)


                frame_data["right_middle"] = (righthand[12].x), (righthand[12].y)
                frame_data["right_middle_knuckle"] = (righthand[9].x), (righthand[9].y)
                frame_data["right_middle_joint"] = (righthand[10].x), (righthand[10].y)


                frame_data["right_ring"] = (righthand[16].x), (righthand[16].y)
                frame_data["right_ring_knuckle"] = (righthand[13].x), (righthand[13].y)
                frame_data["right_ring_joint"] = (righthand[14].x), (righthand[14].y)

                frame_data["right_pinky"] = (righthand[20].x), (righthand[20].y)
                frame_data["right_pinky_knuckle"] = (righthand[17].x), (righthand[17].y)
                frame_data["right_pinky_joint"] = (righthand[18].x), (righthand[18].y)
                                
                for start_idx, end_idx in POSE_CONNECTIONS_HAND:
                    pt1 = righthand[start_idx]
                    pt2 = righthand[end_idx]
                    x1, y1 = int(pt1.x * width), int(pt1.y * height)
                    x2, y2 = int(pt2.x * width), int(pt2.y * height)
                    cv2.line(frame, (x1, y1), (x2, y2), (255, 255, 191), 2)

                    

            cv2.imshow("video", frame)
            delay = 0 if paused else int(1000/ fps)
            key = cv2.waitKey(delay)

            
            if key == ord('q'):
                break

            elif key == ord("p"):
                paused = not paused

            elif key == ord("f"):
                paused = True

            elif key == ord('s'):
                recorded_data.append(frame_data)
                if recorded_data:
                    keys = list(dict.fromkeys(k for row in recorded_data for k in row.keys()))
                    with open("video_pose_data.csv", "w", newline="") as f:
                        writer = csv.DictWriter(f, fieldnames=keys)
                        writer.writeheader()
                        writer.writerows(recorded_data)
                    print("Saved Pose!")


            
                
                
                


            
                
                
                

print(f"Frame rate: {fps}")
print(f"Total Frames: {total_frames}")
cap.release()
cv2.destroyAllWindows()




