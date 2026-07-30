import cv2
import time
import pose_module_IMAGE as pm_image

cap = cv2.VideoCapture('PoseEstimationProject/Poses/6.mp4')
pTime = 0
detector = pm_image.poseDetector()

while True:
    success, img = cap.read()
    if not success:
        break

    def resize_kept_aspect(img, target_width):
        h, w = img.shape[:2]
        scale = target_width / w
        new_height = int(h * scale)
        return cv2.resize(img, (target_width, new_height))
    
    img = resize_kept_aspect(img, 720)

    img = detector.findPose(img)
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