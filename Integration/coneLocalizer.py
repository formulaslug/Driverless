import numpy as np

# Maps YOLO class names to ConeFilter color confidence indices
# ConeFilter order: [blue, yellow, smallOrange, largeOrange]
CLASS_TO_COLOR_INDEX = {
    'seg_blue_cone': 0,
    'seg_yellow_cone': 1,
    'seg_orange_cone': 2,
    'seg_large_orange_cone': 3,
    'seg_unknown_cone': -1
}

def classToColorConfidences(className, detectionConfidence):
    colorIndex = CLASS_TO_COLOR_INDEX.get(className, -1)
    if colorIndex == -1:
        return [0.25, 0.25, 0.25, 0.25]
    confidences = [0.0, 0.0, 0.0, 0.0]
    confidences[colorIndex] = detectionConfidence
    remainder = (1.0 - detectionConfidence) / 3.0
    for i in range(4):
        if i != colorIndex:
            confidences[i] = remainder
    return confidences

def perceptionToDetections(boxes, classes, confidences, coneDistances, cameraIntrinsics):
    # Converts perception output into ConeFilter-format detections
    # Returns list of (xVeh, yVeh, blueConf, yellowConf, sOrangeConf, lOrangeConf)
    if boxes is None or len(boxes) == 0:
        return []

    fx = cameraIntrinsics['fx']
    cx = cameraIntrinsics['cx']

    detections = []
    for i in range(len(boxes)):
        distance = coneDistances[i]['distance']
        if distance is None or distance <= 0 or not np.isfinite(distance):
            continue

        x1, y1, x2, y2 = boxes[i]
        u = (x1 + x2) / 2.0

        # Back-project to vehicle frame (camera Z = vehicle X forward, camera -X = vehicle Y left)
        xVeh = distance
        yVeh = -((u - cx) / fx) * distance

        className = classes[i]['name'] if isinstance(classes[i], dict) else classes[i]
        conf = float(confidences[i]) if confidences is not None else 0.5
        colorConfs = classToColorConfidences(className, conf)

        detections.append((xVeh, yVeh, *colorConfs))

    return detections
