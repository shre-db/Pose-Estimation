import cv2 as cv
import mediapipe as mp
import time


class PoseDetector:
    def __init__(self, mode=False, model_complexity=1, smooth=True, detection_conf=0.5, tracking_conf=0.5):
        self.mode = mode
        self.model_complexity = model_complexity
        self.smooth = smooth
        self.detection_conf = detection_conf
        self.tracking_conf = tracking_conf

        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            self.mode,
            self.model_complexity,
            self.smooth,
            min_detection_confidence=self.detection_conf,
            min_tracking_confidence=self.tracking_conf
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.results = None

    def find_pose(self, img, draw=True):
        img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.results = self.pose.process(img_rgb)
        if self.results.pose_landmarks:
            if draw:
                self.mp_draw.draw_landmarks(img, self.results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)

        return img

    def find_position(self, img, draw=True):
        lm_list = []
        if self.results.pose_landmarks:
            for idx, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                # print(idx, lm)
                cx, cy = int(lm.x*w), int(lm.y*h)
                lm_list.append([idx, cx, cy])
                if draw:
                    cv.circle(img, (cx, cy), 10, (255, 0, 0), cv.FILLED)
        return lm_list


def main():
    cap = cv.VideoCapture(0)
    p_time = 0
    detector = PoseDetector()
    while True:
        success, img = cap.read()
        img = detector.find_pose(img)
        lm_list = detector.find_position(img)
        if lm_list != 0:
            print(lm_list[14])

        c_time = time.time()
        fps = 1/(c_time - p_time)
        p_time = c_time

        cv.putText(img, str(int(fps)), (70, 50), cv.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

        cv.imshow("Image", img)
        cv.waitKey(1)


if __name__ == "__main__":
    main()