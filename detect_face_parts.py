#Written by Ashutosh Gupta

from imutils import face_utils
import numpy as np
import imutils
import dlib
import cv2

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat')

cap = cv2.VideoCapture(0)



part_1 = 'right_eyebrow'
part_2 = 'left_eyebrow'
part_3 = 'nose'

# detect faces in the grayscale image

# loop over the face detections
while 1:
	_, image = cap.read()
	#image = cv2.imread('me.jpg')
	image = imutils.resize(image, width=500)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	rects = detector(gray, 1)
	for (i, rect) in enumerate(rects):
		# determine the facial landmarks for the face region, then
		# convert the landmark (x, y)-coordinates to a NumPy array
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)

		(i,j) = face_utils.FACIAL_LANDMARKS_IDXS[part_1] #right eyebrow index
		(l, m) = face_utils.FACIAL_LANDMARKS_IDXS[part_2] #left eyebrow index
		(y,z) = face_utils.FACIAL_LANDMARKS_IDXS[part_3] #nose index
		
		clone = image.copy() #Cloned image to draw over

		# cv2.putText(clone, part, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
		# 	0.7, (0, 0, 255), 2)

		(a, b, c, d ) = cv2.boundingRect(np.array([shape[y:z]])) #nose coords
		(x, y, w, h) = cv2.boundingRect(np.array([shape[i:j]])) #r-eyebrow coords
		(p, q, r, s) = cv2.boundingRect(np.array([shape[l:m]])) #l-eyebrow coords

		m, n = int((x+w+p)*0.5) , int((y+q)*0.5) #Instatiating starting coords for ROI
		d = int(d*0.75) #To get the height of the Forehead

		#cv2.circle(clone, (x+w, y), 1, (0, 255, 255), -1)
		cv2.line(image, (m, n), (m, n-d), (0,255,0), thickness=1, lineType=8, shift=0) #indicating ROI

		#cv2.rectangle(clone, (x, y), (x + w, y + h), (0, 255, 0), 2)
		# cv2.rectangle(clone, (p, q), (p + r, q + s), (0, 255, 0), 2)
	cv2.imshow("Image", image)	
	if cv2.waitKey(0) & 0xFF == ord('q'):
		break
cap.release()
cv2.destroyAllWindows()