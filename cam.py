#Written by Ashutosh Gupta
import cv2 as cv
import numpy as np

cap = cv.VideoCapture(0)

while True:
	ret, frame = cap.read()
	#gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

	cv.imshow('camera',frame)
	#cv.imshow('gray', gray)

	if cv.waitKey(0) & 0xFF == ord('q'):
		break

cap.release()
cv.destroyAllWindows()