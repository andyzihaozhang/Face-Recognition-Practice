import numpy as np
import cv2
import sys
import dlib
from time import time

import KCF

onDetecting = True
onTracking = False

detector = dlib.get_frontal_face_detector()
tracker = KCF.kcftracker(True, True, False, False)  # hog, fixed_window, multiscale, lab

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if onDetecting:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 0)
        for rect in rects:
            # compute the bounding box of the face and draw it on the frame
            bX, bY, bW, bH = rect.left(), rect.top(), rect.right(), rect.bottom()
            cv2.rectangle(frame, (bX, bY), (bW, bH), (0, 255, 0), 1)

        tracker.init([bX, bY, bW - bX, bH - bY], frame)

        onDetecting = False
        onTracking = True
    elif onTracking:
        boundingbox = tracker.update(frame)  # frame had better be contiguous

        # boundingbox = map(int, boundingbox)
        x1, y1 = boundingbox[0], boundingbox[1]
        x2, y2 = x1 + boundingbox[2], y1 + boundingbox[3]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 1)

    cv2.imshow('tracking', frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cv2.destroyAllWindows()
cap.stop()
