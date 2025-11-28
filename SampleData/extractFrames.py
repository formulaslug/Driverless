import cv2
import os

videoPath = 'driverless.mp4'
outputFolder = 'driverless-10fps'
targetFps = 10

cap = cv2.VideoCapture(videoPath)
videoFps = cap.get(cv2.CAP_PROP_FPS)
frameInterval = int(videoFps / targetFps)

os.makedirs(outputFolder, exist_ok=True)

frameCount = 0
savedCount = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frameCount % frameInterval == 0:
        framePath = os.path.join(outputFolder, f'frame_{savedCount:04d}.jpg')
        cv2.imwrite(framePath, frame)
        savedCount += 1

    frameCount += 1

cap.release()
print('done')
