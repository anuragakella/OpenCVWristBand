import cv2
import numpy as np 
import math
from time import sleep

cap = cv2.VideoCapture(0)
low = np.array([115, 20, 100])
high = np.array([130, 255, 255])
lows = np.array([0, 20, 100])
highs = np.array([20, 255, 255])
font = cv2.FONT_HERSHEY_SIMPLEX

def findLargestContour(contours):
    maxCont = cv2.UMat(np.array([[[0, 0]], [[0, 0]]]))
    maxArea = 50
    maxIn = 0
    i = 0
    for contour in contours:
        c = cv2.contourArea(contour)
        if(c > maxArea):
            maxCont = contour
            maxIn = i
        i += 1
    return maxCont, maxIn

def findSecondLargestContour(contours, cin):
    maxCont = cv2.UMat(np.array([[[0, 0]], [[0, 0]]]))
    maxArea = 50
    contours.pop(cin)
    if(len(contours) != 0):
        i = 0
        for contour in contours:
            c = cv2.contourArea(contour)
            if(c > maxArea):
                maxCont = contour
            i += 1
    return contours, maxCont

def grtSub(a, b):
    if(a > b):
        return int(a - b)
    else:
        return int(b - a)

def EuclidianDistance(x1, y1, x2, y2):
    distance = int(round(math.sqrt(((x1 - x2)**2) + ((y1 - y2)**2))))
    return distance

def invertForeground(mask, x, y, x2, y2):
    for i in range(len(mask)):
            for j in range(len(mask[i])):
                if(j>=x & j<=x2):
                    pass
                else:
                    mask[i][j] = np.array([0, 0, 0])
    return mask


while True: 
    _, img = cap.read()
    img = cv2.flip(img, 1)
    shape = (int(img.shape[0] - 460), int(img.shape[1] - 600))
    shape2 = (int(img.shape[0] - 450), int(img.shape[1] - 600))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv = cv2.GaussianBlur(hsv, (3, 3), 0)
    mask = cv2.inRange(hsv, low, high)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, (20, 20))
    mask2 = cv2.inRange(hsv, lows, highs)
    mask2 = cv2.morphologyEx(mask2, cv2.MORPH_OPEN, (5, 5))
    mask2 = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, (20, 20))
    contours2, hir2 = cv2.findContours(mask2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if(len(contours2) != 0):
        c3 = max(contours2, key=cv2.contourArea)
        x3, y3, w3, h3 = cv2.boundingRect(c3)
        # mask2 = mask2[y3:(y3+h3), x3:(x3+w3)]

    contours, hir = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if(len(contours) != 0):
        cin, index = findLargestContour(contours)
        cont2, cin2 = findSecondLargestContour(contours, index)
        x, y, w, h = cv2.boundingRect(cin)
        x2, y2, w2, h2 = cv2.boundingRect(cin2)
        center_c1 = (int((x+(w/2))), int((y+(h/2))))
        center_c2 = (int((x2+(w2/2))), int((y2+(h2/2))))
        pix_center = (int(min(center_c1[0], center_c2[0]) + grtSub(center_c1[0], center_c2[0])/2), int(max(center_c1[1], center_c2[1]) - grtSub(center_c1[1], center_c2[1])/2))
        if(pix_center[0] == 0):
            cv2.putText(img, 'waiting for track points', shape, font, 1, (0, 0, 0), 1, cv2.LINE_AA)
        if(((x != 0) & (x2 != 0)) & ((y != 0) & (y2 != 0))):
            if(((center_c2[1]/480 - center_c1[1]/480) != 0) & ((center_c2[0]/640 - center_c1[0]/640) != 0)):
                sleep(0.05)
                gradient = float(center_c2[1]- center_c1[1])/float(center_c2[0] - center_c1[0])
                angle = (np.arctan(gradient)*(180/math.pi))
                if(angle < 0):
                    angle = 180 + angle
            else:
                angle = 90
            cv2.putText(img, str(int(angle)), pix_center, font, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 0), 1)
            cv2.putText(img, '1', (x, y), font, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.circle(img, center_c1, 1, (255, 0, 255), -1)
            cv2.circle(img, center_c2, 1, (255, 0, 255), -1)
            rx = pix_center[0] - 100
            ry = pix_center[1] + 180
            r2x = pix_center[0] + 100
            r2y = pix_center[1] - 180
            cv2.rectangle(img, (x3, y3), (x3+ w3, y3 + h3), (0, 0, 0), 1)
            mask2 = invertForeground(mask2, x3, y3, (x3+h), (w3+h))
            moments = cv2.moments(c3)
            cx = int(moments['m10']/moments['m00'])
            cy = int(moments['m01']/moments['m00'])
            handc = (cx, cy)
            print(handc)
            dist = EuclidianDistance(cx, cy, pix_center[0], pix_center[1])
            buttonPressCenter = (int(min(pix_center[0], cx) + grtSub(pix_center[0], cx)/2), int(max(pix_center[1], cy) - grtSub(pix_center[1], cy)/2))
            if(dist > 110):
                cv2.putText(img, '0', shape, font, 1, (0, 0, 0), 1, cv2.LINE_AA)
            elif(dist <= (dist*1.5)):
                cv2.putText(img, '1', shape, font, 1, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(img, str(dist), buttonPressCenter, font, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(img, '2', (x2, y2), font, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.circle(img, handc, 1, (255, 100, 255), -1)
            cv2.circle(img, pix_center, 1, (255, 100, 255), -1)
            cv2.line(img, pix_center, handc, (255, 255, 255), 1)
            cv2.rectangle(img, (x2, y2), (x2 + w2, y2 + h2), (0, 0, 0), 1)
            cv2.line(img, center_c1, center_c2, (0, 0, 0), 1)
            cv2.putText(img, '2', (x2, y2), font, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    
    cv2.imshow('hsv', hsv)
    cv2.imshow('mask', mask)
    cv2.imshow('f', img)
    cv2.imshow('m2', mask2)
  
    if(cv2.waitKey(1) == 27):
        break

cap.release()
cv2.destroyAllWindows()
