import math
import cv2
import numpy as np
from time import time
import mediapipe as mp
import matplotlib.pyplot as pit

cap = cv2.VideoCapture('assets/dance.mp4')


while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        else:
            fps= cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        

        frame = cv2.flip(frame, 1)

        

        cv2.imshow("webcam", frame)
        if cv2.waitKey(int(1000/ 30)) == ord('q'):
            break

print(f"Frame rate: {fps}")
print(f"Total Frames: {total_frames}")
cap.release()
cv2.destroyAllWindows()




