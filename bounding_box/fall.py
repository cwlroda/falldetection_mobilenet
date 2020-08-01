#Adapted from Zed code - Person Fall detection using raspberry pi camera and opencv lib.
#Adapted from pyimage research
#Adapted from python object detection

import cv2
import time

fitToEllipse = False
cap = cv2.VideoCapture("/home/htxsns/projects/gettyimages-458529130-640_adpp.mp4")
time.sleep(2)

fgbg = cv2.createBackgroundSubtractorMOG2()
j = 0

#For each frame readed of the video corverted into gray, is removed the background, finded the contour and drawed the contours.
#If the heigh of the contour is lower than width, it may be a fall and we add 1 to a count, if the count is greater than 10, will be drawed a rectangle to the possible person fallen.

while(1):
    #captures frames in the video
    ret, frame = cap.read()

    # describe the type of font
    # to be used.
    font = cv2.FONT_HERSHEY_SIMPLEX

    #Convert each frame to gray scale and subtract the background
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        fgmask = fgbg.apply(gray)

        #Find contours
        contours, _ = cv2.findContours(fgmask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # List to hold all areas
            areas = []

            for contour in contours:
                ar = cv2.contourArea(contour)
                areas.append(ar)

            max_area = max(areas, default = 0)
            max_area_index = areas.index(max_area)
            cnt = contours[max_area_index]
            M = cv2.moments(cnt)

            x, y, w, h = cv2.boundingRect(cnt)
            cv2.drawContours(fgmask, [cnt], 0, (255,255,255), 3, maxLevel = 0)

            if h < w:
                j += 1

            if j > 10:
                #print("FALL")
                cv2.putText(frame, 'FALL', (x, y-10), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (0,0,255), 2)
                cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)

            if h > w:
                j = 0
                cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

            cv2.imshow('video', frame)

            if cv2.waitKey(33) == 27:
                break

    except Exception as e:
        break

cv2.destroyAllWindows()