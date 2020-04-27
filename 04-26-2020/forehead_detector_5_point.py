# USAGE
# python forehead_detector_5_point.py shape_predictor_5_face_landmarks.dat

# import the necessary packages
import sys
import time
import dlib
import cv2
import numpy as np

# construct the argument parser and parse the arguments
predictor_path = sys.argv[1]

# initialize dlib's face detector (HOG-based) and then create the
# facial landmark predictor
print("[INFO] loading facial landmark predictor...")
print("[INFO] loading HOG algorithm...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

# initialize the video stream and sleep for a bit, allowing the
# camera sensor to warm up
vs = cv2.VideoCapture(0)

# initialize the loop and fps list for average fps estimation
loop = 0
fps = []


def get_forehead_coord(face_landmarks):
    left = [face_landmarks.part(2).x, face_landmarks.part(2).y]
    right = [face_landmarks.part(0).x, face_landmarks.part(0).y]
    nose = [face_landmarks.part(4).x, face_landmarks.part(4).y]

    eye_center = [(left[0]+right[0])/2, (left[1]+right[1])/2]
    # distance between nose and eye center
    dist1 = [nose[0]-eye_center[0], nose[1]-eye_center[1]]
    # distance between eye center to left eye
    dist2 = [eye_center[0]-left[0], eye_center[1]-left[1]]
    bottom_center = [eye_center[0]-2/3*dist1[0], eye_center[1]-2/3*dist1[1]]
    left_bottom = [bottom_center[0]-dist2[0], bottom_center[1]-dist2[1]]
    right_bottom = [bottom_center[0]+dist2[0], bottom_center[1]+dist2[1]]
    left_top = [left_bottom[0]-dist1[0], left_bottom[1]-dist1[1]]
    right_top = [right_bottom[0]-dist1[0], right_bottom[1]-dist1[1]]

    coord = np.array([left_bottom, right_bottom, right_top, left_top], np.int32)
    coord = [coord.reshape((-1, 1, 2))]
    return coord


# loop over the frames from the video stream
while True:
    # start timer
    start = time.time()
    
    # grab the frame from the threaded video stream
    ret, frame = vs.read()

    # recognize face every 5 frames, otherwise use most recent coordinates
    if loop % 5 == 0:
        # convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # detect faces in the grayscale frame
        rects = detector(gray, 0)

        # check to see if a face was detected, and if so, draw the total
        # number of faces on the frame
        if len(rects) > 0:
            text = "{} face(s) found".format(len(rects))
            cv2.putText(frame, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 255), 2)

        forehead_coord_list = []
        # loop over the face detections
        for rect in rects:
            # compute the bounding box of the face and draw it on the frame
            bX, bY, bW, bH = rect.left(), rect.top(), rect.right(), rect.bottom()
            cv2.rectangle(frame, (bX, bY), (bW, bH), (0, 255, 0), 1)

            # determine the facial landmarks for the face region
            shape = predictor(gray, rect)

            # store forehead coord for frame do not run face recognition
            forehead_coord_list.append(get_forehead_coord(shape))

            # draw the bounding box of the forehead
            cv2.polylines(frame, get_forehead_coord(shape), True, (0, 255, 255))

    else:
        if len(rects) > 0:
            text = "{} face(s) found".format(len(rects))
            cv2.putText(frame, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, (0, 0, 255), 2)

        for rect in rects:
            bX, bY, bW, bH = rect.left(), rect.top(), rect.right(), rect.bottom()
            cv2.rectangle(frame, (bX, bY), (bW, bH), (0, 255, 0), 1)

        for forehead_coord in forehead_coord_list:
            cv2.polylines(frame, forehead_coord, True, (0, 255, 255))

    # show the frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    loop += 1

    # end timer
    end = time.time()

    # calculate the fps and current frame and add it to fps list for
    # average fps estimation
    frame_fps = 1/(end - start)
    fps.append(frame_fps)
    
    print("Average fps: ", sum(fps)/loop)
 
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
