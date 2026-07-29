import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_styles
from mediapipe.tasks.python.vision import drawing_utils
import time

mpDraw = vision
base_options = python.BaseOptions(model_asset_path = 'PoseEstimationProject/pose_landmarker_full.task')
options = vision.PoseLandmarkerOptions(base_options = base_options)
detector = vision.PoseLandmarker.create_from_options(options)

cap = cv2.VideoCapture('PoseEstimationProject/Poses/1.mp4')

pTime = 0

while True:
    success, img = cap.read()
    

    if not success:
        break

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (720, 1080))
    mp_image = mp.Image(image_format = mp.ImageFormat.SRGB, data = imgRGB)
    detection_result = detector.detect(mp_image)

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    # print(detection_result)
    if detection_result.pose_landmarks:
        for pose_landmarks in detection_result.pose_landmarks:
            drawing_utils.draw_landmarks(
                image = img,
                landmark_list = pose_landmarks,
                connections = vision.PoseLandmarksConnections.POSE_LANDMARKS
            )
            for id, lm in enumerate(pose_landmarks):
                h, w, c = img.shape
                cx, cy =  int(lm.x * w), int(lm.y * h)
                print (f"Landmark = {id} x = {lm.x:.2f} y = {lm.y:.2f} visibility = {lm.visibility:.2f}")
                cv2.circle(img, (cx, cy), 5, (0, 0, 0), cv2.FILLED)


                


    cv2.putText (img, str(int(fps)), (30, 40), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    cv2.imshow("Image", img)

    cv2.waitKey(1)


   
