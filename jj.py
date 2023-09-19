# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import cv2
import mediapipe as mp
import time

class poseDetector():
    def __init__(self, mode= False,upBody=False,smppth=True,detectionCon= 0.5,trackCon=0.5):
        self.mode = mode
        self.upBody = upBody
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose= mp.solutions.pose
        self.pose= self.mpPose.Pose(self.mode,self.upBody,self.smooth,self.detectionCon,self.trackCon)

    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColpr(img, cv2.COLOR_BGR2RGB)
        results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
        return img
def findPosition(self, img, draw=True):
    lmlist = []
    if self.results.pose_landmarks:
        for id, lm in enumerate(self.results.pose_landmarks.landmark):
             h, w, c = img.shape
             #print(id, lm)
             cx, cy = int(lm.x * w), int(lm.y * h)
             lmlist.append([id, cx, cy])
             if draw:
                  cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        return lmlist
def main():

    cap= cv2.VideoCApture('PoseVedios/1.mp4')
    pTime = 0
    detector = poseDetector()
    while True:
        success, img = cap.read()
        img =detector.findPose(img)
        lmList = detector.findPosition(img,draw=False)
        print(lmList[14])
        cv2.circle(img, (lmList[14][1],lmList[14][2]), 5, (0, 0, 225), cv2.FILLED)


        cTime = time.time()
        fps= 1/(cTime-pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (70,50),cv2.FONT_HERSHEY_PLAIN,3,(255,0,0),3)

        cv2.imshow("Image",img)
        cv2.waitKey(1)

