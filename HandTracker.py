'''
Name: Ali Sartaz Khan
Description:Track the left hand of an individual using the front facing camera and correctly identify 
any number from 1-10 represented using Sign Language. 
'''

import os
import cv2
import math
import mediapipe as mp

class handDetector():
    '''
    Class that uses the front camera and detects all the 22 points in the person's finger.
    '''
    def __init__(self, mode=False, maxHands=1, modelComplex = 1, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.modelComplex = 1
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.modelComplex, self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        '''
        Functions detects the hand and draws dots on all 22 points of the hand.
        
        img: One frame obtained from the front camera of the laptop
        '''
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms,
                                               self.mpHands.HAND_CONNECTIONS)
        return img
    def findPosition(self, img, handNo=0, draw=True):
        '''
        Find position of fingers and gives their coordinates
        
        img: 1 frame obtained from the front camera of the laptop
        '''
        lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
        return lmList


def imageList():
    '''
    Function returns a list of images is the FingerImages directory.
    '''
    folderPath = "FingerImages"
    myList = os.listdir(folderPath)
    overlayList = []

    for imgPath in myList:
        image = cv2.imread(f'{folderPath}/{imgPath}')
        overlayList.append(image)

    return overlayList


def checkFingers(fingers, img, overlayList, lmList):
    '''
    Checks if the fingers are in a particular orientation that matches the ASL representations 1-10.
    If so, it displays the appropriate number image representated by the person.
    
    fingers: List of 0s and 1s for 5 fingers starting from the thumb to the pinky where 1 means finger is up and 0
             means finger is down
    img: One frame obtained from front camera
    overlayList: List containing all the pictures of ASL gestures 1-10
    lmList: 2D list of all fingers and their coordinates
    '''
    if fingers == [0, 1, 0, 0, 0]: # one
        displayImage(img, overlayList[0])
    elif fingers == [0, 1, 1, 0, 0]: #two
        displayImage(img, overlayList[1])
    elif fingers == [1, 1, 1, 0, 0]: #three
        displayImage(img, overlayList[2])
    elif fingers == [0, 1, 1, 1, 1]: #four
        displayImage(img, overlayList[3])
    elif fingers == [1, 1, 1, 1, 1]: #five
        displayImage(img, overlayList[4])
    elif fingers == [0, 1, 1, 1, 0] and isTouching(lmList, 20, 4): #six
        displayImage(img, overlayList[5])
    elif fingers == [0, 1, 1, 0, 1] and isTouching(lmList, 16, 4): #seven
        displayImage(img, overlayList[6])
    elif fingers == [1, 1, 0, 1, 1] and isTouching(lmList, 12, 4): #eight
        displayImage(img, overlayList[7])
    elif fingers == [1, 0, 1, 1, 1] and isTouching(lmList, 8, 4): #nine
        displayImage(img, overlayList[8])
    elif fingers == [1, 0, 0, 0, 0]: #ten
        displayImage(img, overlayList[9])

def displayImage(img, image):
    '''
    Displays the Image
    img: One frame obtained from the front camera
    image: Image to be displayed
    '''
    height, width, channel = image.shape
    img[0:height, 0:width] = image



def isTouching(lmList, finger1, finger2):
    '''
    Function checks if the two fingers in the parameters are touching and returns true or false.
    
    lmList: 2D list of all fingers and their coordinates
    finger1: ID number of the finger
    finger2: ID number of the finger
    '''
    x1, y1 = lmList[finger1][1], lmList[finger1][2]
    x2, y2 = lmList[finger2][1], lmList[finger2][2]
    length = math.hypot(x2 - x1, y2 - y1)
    if length < 50:
        return True
    return False

def main():
    wCap, hCap = 640, 480
    cap = cv2.VideoCapture(0)
    cap.set(3,wCap)
    cap.set(4,hCap)

    overlayList = imageList()
    detector = handDetector(detectionCon=-.75)
    tipIds = [4, 8, 12, 16, 20]

    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        lmList = detector.findPosition(img, draw = False)
        if len(lmList) != 0:
            fingers = []
            for id in range(5):
                if id == 0:
                    if lmList[tipIds[id]][1] > lmList[tipIds[id]-2][1]:
                        fingers.append(0)
                    else:
                        fingers.append(1)
                elif lmList[tipIds[id]][2] < lmList[tipIds[id]-2][2]:
                    fingers.append(1)
                else:
                    fingers.append(0)
            checkFingers(fingers, img, overlayList, lmList)
        cv2.imshow("Image", img)
        cv2.waitKey(1)

main()
