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

def perceptionToDetections(boxes, classes, confidences, coneDistances):
    # Converts perception output into ConeFilter-format detections
    # Returns list of (xVeh, yVeh, blueConf, yellowConf, sOrangeConf, lOrangeConf)
    if boxes is None or len(boxes) == 0:
        return []

    detections = []
    for i in range(len(boxes)):
        x, y = coneDistances[i]['x'], coneDistances[i]['y']
        if x is None or y is None or x <= 0 or not np.isfinite(x) or not np.isfinite(y):
            continue

        className = classes[i]['name'] if isinstance(classes[i], dict) else classes[i]
        conf = float(confidences[i]) if confidences is not None else 0.5
        colorConfs = classToColorConfidences(className, conf)

        detections.append((x, y, *colorConfs))

    return detections

def coneFilterToPathPlannerInput(cones):
    if cones is None or len(cones) < 4:
        return None

    positions = np.array([[cone.x, cone.y] for cone in cones])
    coordinateUncertainty = np.array([np.sqrt(np.trace(cone.cov) / 2) for cone in cones])
    colors = np.array([cone.colorConfidences for cone in cones])

    return positions, coordinateUncertainty, colors
