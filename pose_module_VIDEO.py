import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
# from mediapipe.tasks.python.vision import drawing_styles
from mediapipe.tasks.python.vision import drawing_utils
import time
import random

class poseDetector():
    def __init__(self, model_path = 'PoseEstimationProject/pose_landmarker_full.task'):

        # self.count_ids = set()
        self.color_detect = {}
        
        self.model_path = model_path
        base_options = python.BaseOptions(model_asset_path = self.model_path)
        options = vision.PoseLandmarkerOptions(
            base_options = base_options,
            running_mode = vision.RunningMode.VIDEO,
            num_poses = 6,
            min_pose_detection_confidence = 0.55,
            min_pose_presence_confidence = 0.7,
            min_tracking_confidence = 0.75
            )
        self.detector = vision.PoseLandmarker.create_from_options(options)
        self.mpDraw = drawing_utils

    def get_color(self, person_idx):
        if person_idx not in self.color_detect:
            color_1 = random.randint(0, 255)
            color_2 = random.randint(0, 255)
            color_3 = random.randint(0, 255)
            new_color = (color_1, color_2, color_3)
            while new_color in self.color_detect.values():
                color_1 = random.randint(0, 255)
                color_2 = random.randint(0, 255)
                color_3 = random.randint(0, 255)
                new_color = (color_1, color_2, color_3)
            self.color_detect[person_idx] = new_color

        return self.color_detect[person_idx]
            

    def findPose(self, img, fps_timestamp_ms, draw = True):

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        mp_image = mp.Image(image_format = mp.ImageFormat.SRGB, data = imgRGB)
        self.detection_result = self.detector.detect_for_video(mp_image, fps_timestamp_ms)

        if self.detection_result.pose_landmarks:
            if draw:
                for person_idx, pose_landmarks in enumerate (self.detection_result.pose_landmarks):
                    color = self.get_color(person_idx)
                    connection_spec = drawing_utils.DrawingSpec(color = color,
                                                                thickness = 2)
                    drawing_utils.draw_landmarks(
                        image = img,
                        landmark_list = pose_landmarks,
                        connections = vision.PoseLandmarksConnections.POSE_LANDMARKS,
                        connection_drawing_spec = connection_spec
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