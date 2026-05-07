
import cv2
import mediapipe as mp
import  time

#
# from mediapipe.python.solutions import hands as mp_hands
# from mediapipe.python.solutions import drawing_utils as mp_drawing

class detectHand():

    def __init__(self, mode=False, maxHands = 2, detectionCon = 0.5, trackCon = 0.5):

        # static_image_mode = False,
        # max_num_hands = 2,
        # min_detection_confidence = 0.5,
        # min_tracking_confidence = 0.5

        # self.mpDraw = mp_drawing
        # self.mpHands = mp_hands

        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = float(detectionCon)
        self.trackCon = float(trackCon)

        # Creating object for handdetection module

        self.mpHands = mp.solutions.hands  # default line to initiate the module

        #  Calling default hand function
        self.hands = self.mpHands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.maxHands,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon
        )

        #  drawing the landmarks of the hand points using default function
        self.mpDraw = mp.solutions.drawing_utils

    def detect_Hand_lmks(self, img, draw = True ):

        #   Converting to img to RGB scale, as hands use only RGB images by default
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        #  Processing the img
        self.res = self.hands.process(imgRGB)

        #  Open the object extract the multiple hands ancd check if the hands are detected
        # print(res.multi_hand_landmarks)

        if  self.res.multi_hand_landmarks:

            for handsLmks in self.res.multi_hand_landmarks:

                # function to dected points of hand
                # for id, lmk in enumerate( handsLmks.landmark):

               if(draw):
                #  Displays the points and connections of the hand fingures
                    self.mpDraw.draw_landmarks(img, handsLmks, self.mpHands.HAND_CONNECTIONS)

        return img

    def detectPosition(self, img, handNum = 0, draw = True):

        lmkList = []
        if self.res.multi_hand_landmarks :

            myHand = self.res.multi_hand_landmarks[handNum]

            for id, lmks in enumerate(myHand.landmark):
                # print(id, lmk)

                h, w, c = img.shape
                cx, cy = int(lmks.x * w), int(lmks.y * h)
                # print(id, cx, cy)

                # if id == 0: # ( highlights palm )
                #     cv2.circle(img,( cx, cy), 15, (255, 0, 255), cv2.FILLED)

                lmkList.append(([id , cx, cy] ) );

                if draw:
                    #  highlighting the 21 points of hand
                    cv2.circle(img, (cx, cy), 9, (255, 0, 255), cv2.FILLED)

        return lmkList

def main():
    cap = cv2.VideoCapture(0)

    detectorObj = detectHand()

    pTime = 0
    cTime = 0

    while True:

        success, img = cap.read()
        img = detectorObj.detect_Hand_lmks(img)
        lmList = detectorObj.detectPosition(img)

        if len(lmList) != 0:
            #  print the point only at index 4
            print(lmList[4])

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        #  store the 21 points of hand in a list
        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

        cv2.imshow("Image", img)

        cv2.waitKey(1)



if __name__ == "__main__":
    main()
