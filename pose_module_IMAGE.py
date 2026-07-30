import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
# from mediapipe.tasks.python.vision import drawing_styles
from mediapipe.tasks.python.vision import drawing_utils
import time

class poseDetector():
    def __init__(self, model_path = 'PoseEstimationProject/pose_landmarker_full.task'):
        self.model_path = model_path
        base_options = python.BaseOptions(model_asset_path = self.model_path)
        options = vision.PoseLandmarkerOptions(base_options = base_options)
        self.detector = vision.PoseLandmarker.create_from_options(options)

        self.mpDraw = drawing_utils


    def findPose(self, img, draw = True):

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        mp_image = mp.Image(image_format = mp.ImageFormat.SRGB, data = imgRGB)
        self.detection_result = self.detector.detect(mp_image)

        if self.detection_result.pose_landmarks:
            if draw:
                for pose_landmarks in self.detection_result.pose_landmarks:
                    drawing_utils.draw_landmarks(
                        image = img,
                        landmark_list = pose_landmarks,
                        connections = vision.PoseLandmarksConnections.POSE_LANDMARKS
                    )
        return img

    

    def findPosition(self, img, draw = True):

        lmList = []
        if self.detection_result.pose_landmarks:
            for pose_landmarks in self.detection_result.pose_landmarks:
                for id, lm in enumerate(pose_landmarks):
                    h, w, c = img.shape
                    cx, cy =  int(lm.x * w), int(lm.y * h)
                    lmList.append([id, cx, cy])
                    if draw:
                        # print (f"Landmark = {id} x = {lm.x:.2f} y = {lm.y:.2f} ")
                        cv2.circle(img, (cx, cy), 3, (0, 0, 255), cv2.FILLED)

        return lmList


# def main():
