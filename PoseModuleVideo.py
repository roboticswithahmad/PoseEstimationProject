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
        options = vision.PoseLandmarkerOptions(
            base_options = base_options,
            running_mode = vision.RunningMode.VIDEO,
            num_poses = 2,
            min_pose_detection_confidence = 0.6,
            min_pose_presence_confidence = 0.6,
            min_tracking_confidence = 0.6
            )
        self.detector = vision.PoseLandmarker.create_from_options(options)
        self.mpDraw = drawing_utils


    def findPose(self, img, fps_timestamp_ms, draw = True):

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        mp_image = mp.Image(image_format = mp.ImageFormat.SRGB, data = imgRGB)
        self.detection_result = self.detector.detect_for_video(mp_image, fps_timestamp_ms)

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


def main():
    

    cap = cv2.VideoCapture('PoseEstimationProject/Poses/3.mp4')

    pTime = 0

    detector = poseDetector()
    def resize_kept_aspect(img, target_width):
                h, w = img.shape[:2]
                scale = target_width / w
                new_height = int(h * scale)
                return cv2.resize(img, (target_width, new_height))

    while True:
        success, img = cap.read()
        if not success:
            break

        img = resize_kept_aspect(img, 720)
        timestamp_ms = int(time.time() * 1000)
        img = detector.findPose(img, timestamp_ms)
        lmList = detector.findPosition(img)
        if len(lmList) != 0:
            print(lmList)

        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime
        cv2.putText (img, str(int(fps)), (30, 40), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

        cv2.imshow("Image", img)

        if cv2.waitKey(1) & 0xFF == ord ('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

