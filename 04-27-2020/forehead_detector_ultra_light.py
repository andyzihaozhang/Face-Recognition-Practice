"""
Required Packages and Versions:
onnx==1.6.0
onnx-tf==1.3.0
onnxruntime==0.5.0
opencv-python==4.1.1.26
tensorflow==1.13.1

Model Used:
detector = ultra_light_320.onnx
predictor = shape_predictor_5_face_landmarks.dat
"""

import time
import dlib
import cv2
import numpy as np
import onnx
import onnxruntime as ort
from onnx_tf.backend import prepare


def area_of(left_top, right_bottom):
    """
    Compute the areas of rectangles given two corners.
    Args:
        left_top (N, 2): left top corner.
        right_bottom (N, 2): right bottom corner.
    Returns:
        area (N): return the area.
    """
    hw = np.clip(right_bottom - left_top, 0.0, None)
    return hw[..., 0] * hw[..., 1]


def iou_of(boxes0, boxes1, eps=1e-5):
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Args:
        boxes0 (N, 4): ground truth boxes.
        boxes1 (N or 1, 4): predicted boxes.
        eps: a small number to avoid 0 as denominator.
    Returns:
        iou (N): IoU values.
    """
    overlap_left_top = np.maximum(boxes0[..., :2], boxes1[..., :2])
    overlap_right_bottom = np.minimum(boxes0[..., 2:], boxes1[..., 2:])

    overlap_area = area_of(overlap_left_top, overlap_right_bottom)
    area0 = area_of(boxes0[..., :2], boxes0[..., 2:])
    area1 = area_of(boxes1[..., :2], boxes1[..., 2:])
    return overlap_area / (area0 + area1 - overlap_area + eps)


def hard_nms(box_scores, iou_threshold, top_k=-1, candidate_size=200):
    """
    Perform hard non-maximum-supression to filter out boxes with iou greater
    than threshold
    Args:
        box_scores (N, 5): boxes in corner-form and probabilities.
        iou_threshold: intersection over union threshold.
        top_k: keep top_k results. If k <= 0, keep all the results.
        candidate_size: only consider the candidates with the highest scores.
    Returns:
        picked: a list of indexes of the kept boxes
    """
    scores = box_scores[:, -1]
    boxes = box_scores[:, :-1]
    picked = []
    indexes = np.argsort(scores)
    indexes = indexes[-candidate_size:]
    while len(indexes) > 0:
        current = indexes[-1]
        picked.append(current)
        if 0 < top_k == len(picked) or len(indexes) == 1:
            break
        current_box = boxes[current, :]
        indexes = indexes[:-1]
        rest_boxes = boxes[indexes, :]
        iou = iou_of(
            rest_boxes,
            np.expand_dims(current_box, axis=0),
        )
        indexes = indexes[iou <= iou_threshold]

    return box_scores[picked, :]


def predict(width, height, confidences, boxes, prob_threshold, iou_threshold=0.5, top_k=-1):
    """
    Select boxes that contain human faces
    Args:
        width: original image width
        height: original image height
        confidences (N, 2): confidence array
        boxes (N, 4): boxes array in corner-form
        iou_threshold: intersection over union threshold.
        top_k: keep top_k results. If k <= 0, keep all the results.
    Returns:
        boxes (k, 4): an array of boxes kept
        labels (k): an array of labels for each boxes kept
        probs (k): an array of probabilities for each boxes being in corresponding labels
    """
    boxes = boxes[0]
    confidences = confidences[0]
    picked_box_probs = []
    picked_labels = []
    for class_index in range(1, confidences.shape[1]):
        probs = confidences[:, class_index]
        mask = probs > prob_threshold
        probs = probs[mask]
        if probs.shape[0] == 0:
            continue
        subset_boxes = boxes[mask, :]
        box_probs = np.concatenate([subset_boxes, probs.reshape(-1, 1)], axis=1)
        box_probs = hard_nms(box_probs, iou_threshold=iou_threshold, top_k=top_k)
        picked_box_probs.append(box_probs)
        picked_labels.extend([class_index] * box_probs.shape[0])
    if not picked_box_probs:
        return np.array([]), np.array([]), np.array([])
    picked_box_probs = np.concatenate(picked_box_probs)
    picked_box_probs[:, 0] *= width
    picked_box_probs[:, 1] *= height
    picked_box_probs[:, 2] *= width
    picked_box_probs[:, 3] *= height
    return picked_box_probs[:, :4].astype(np.int32), np.array(picked_labels), picked_box_probs[:, 4]


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


video_capture = cv2.VideoCapture(0)

# onnx_path = 'ultra_light_640.onnx'
onnx_path = 'ultra_light_320.onnx'
predictor_path = 'shape_predictor_5_face_landmarks.dat'
onnx_model = onnx.load(onnx_path)
detector = prepare(onnx_model)
predictor = dlib.shape_predictor(predictor_path)
ort_session = ort.InferenceSession(onnx_path)
input_name = ort_session.get_inputs()[0].name

loop = 0
fps = []

while True:
    # start timer
    start = time.time()

    ret, frame = video_capture.read()
    h, w, _ = frame.shape

    # preprocess img acquired
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # convert bgr to rgb
    # img = cv2.resize(img, (640, 480))  # resize to 640 * 480
    img = cv2.resize(img, (320, 240))  # resize to 320 * 240
    img_mean = np.array([127, 127, 127])
    img = (img - img_mean) / 128
    img = np.transpose(img, [2, 0, 1])
    img = np.expand_dims(img, axis=0)
    img = img.astype(np.float32)

    confidences, boxes = ort_session.run(None, {input_name: img})
    boxes, labels, probs = predict(w, h, confidences, boxes, 0.7)

    for i in range(boxes.shape[0]):
        box = boxes[i, :]
        x1, y1, x2, y2 = box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (80, 18, 236), 2)
        cv2.rectangle(frame, (x1, y2 - 20), (x2, y2), (80, 18, 236), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        text = f"face: {labels[i]}"
        cv2.putText(frame, text, (x1 + 6, y2 - 6), font, 0.5, (255, 255, 255), 1)

        rect = dlib.rectangle(left=x1, top=y1, right=x2, bottom=y2)
        shape = predictor(gray, rect)
        cv2.polylines(frame, get_forehead_coord(shape), True, (0, 255, 255))

    cv2.imshow('Video', frame)

    loop += 1

    # end timer
    end = time.time()

    # calculate the fps and current frame and add it to fps list for
    # average fps estimation
    frame_fps = 1 / (end - start)
    fps.append(frame_fps)

    print("Average fps: ", sum(fps) / loop)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
