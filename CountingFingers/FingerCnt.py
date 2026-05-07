
import cv2

import mediapipe as mp

import time

import os
import HndTrackingModule  as HTM

# webcam

widthCam, heightCam = 650, 500

cap = cv2.VideoCapture(0)
cap.set(3, widthCam)
cap.set(4, heightCam )

folderPth = 'Images'
# data = os.listdir(folderPth)

data = [f for f in os.listdir(folderPth)
        if f.endswith(('.png', '.jpg', '.jpeg'))]
data.sort() # Ensures 0.jpeg is index 0, 1.jpeg is index 1, etc.
print(data)

# overlaying list of images

overlayList = []

for imgPth in data:

    image = cv2.imread(f'{folderPth}/{imgPth}')

    print(f'{folderPth}/{imgPth}')

#      saving in list
    overlayList.append(image)

print(len(overlayList))

pTime = 0

detector = HTM.detectHand(detectionCon = 0.75)

lmkTipIds = [4, 8, 12, 16, 20]

while True:

    success, img = cap.read()

    if not success:
        continue

    img = detector.detect_Hand_lmks(img)

    #  list of lmks
    lmkList = detector.detectPosition(img, draw = False)

    # print(lmkList)

    if len(lmkList) != 0:
        #  lmk points of finger tip to count the number of fingers
        #  index finger point 8 y-axis value < point 6 y-axis value  ==> open
        # if lmkList[8][2] < lmkList[6][2]:
        #     print("Index is open")
        # checking for all fingers

        fingers=[]

        #  for thumb
        if lmkList[lmkTipIds[0]][1] > lmkList[lmkTipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        for id in range(1, 5):

            #  for fingers (index to pinky)
            if lmkList[lmkTipIds[id]][2] < lmkList[lmkTipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        # print (fingers)

        #  calculating the number of fingers
        totalFingers = fingers.count(1)
        print(f"Finger detected: ", totalFingers)


        #  putting overlay img at 0th position
        # [width, height] of image

        if( totalFingers < len(overlayList)):
            # ht, wd, c = overlayList[totalFingers].shape
            # img[0:ht, 0:wd] = overlayList[totalFingers]
            imgOverlay = overlayList[totalFingers]
            ht, wd, c = imgOverlay.shape

            # Ensure overlay fits within the webcam frame
            img[0:ht, 0:wd] = imgOverlay

    #  calculating FPS
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, f' FPS: {int(fps)}', (400, 60), cv2.FONT_HERSHEY_COMPLEX, 3, (255, 0, 0), 3)


    cv2.imshow("FingersCounting", img)
    # cv2.waitKey(1)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Added a way to break the loop
        break

cap.release()
cv2.destroyAllWindows()












