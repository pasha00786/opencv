# Written by Ashutosh Gupta
# Pyhton code to generate a funky frame over the human face using OpenCV and dlib library.

from imutils import face_utils
import numpy as np
import imutils
import dlib
import cv2

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat') #Loading the facial HOG Detector

cap = cv2.VideoCapture(0) #Getting frames from webcam.

#Instantiating facial index
part_1 = 'right_eyebrow'
part_2 = 'left_eyebrow'
part_3 = 'nose'
part_4 = 'jaw'


while 1:  #to continously stream the frames
	_, image = cap.read()
	#image = cv2.imread('me.jpg')
	image = imutils.resize(image, width=500)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #Grayscaled image to have fast and efficient detection
	rects = detector(gray, 1)
	for (i, rect) in enumerate(rects):
		# determine the facial landmarks for the face region, then                 
		shape = predictor(gray, rect)
		# convert the landmark (x, y)-coordinates to a NumPy array
		shape = face_utils.shape_to_np(shape)

		#Extracting specific facial points as tuple
		(i,j) = face_utils.FACIAL_LANDMARKS_IDXS[part_1] #right eyebrow index
		(l, m) = face_utils.FACIAL_LANDMARKS_IDXS[part_2] #left eyebrow index
		(y,z) = face_utils.FACIAL_LANDMARKS_IDXS[part_3] #nose index
		(k,e) = face_utils.FACIAL_LANDMARKS_IDXS[part_4] #jawline index

		#Getting bounding boxes coords
		(a, b, c, d) = cv2.boundingRect(np.array([shape[y:z]])) #nose coords
		(x, y, w, h) = cv2.boundingRect(np.array([shape[i:j]])) #r-eyebrow coords
		(p, q, r, s) = cv2.boundingRect(np.array([shape[l:m]])) #l-eyebrow coords
		(u, i, o, z) = cv2.boundingRect(np.array([shape[k:e]])) #jaw-line coords

		#Calculating the originating coords and height of the forehead ROI 
		m, n = int((x+w+p)*0.5) , int((y+q)*0.5)
		d = int(d*0.75)

		#Overlapping ROI over the frame
		cv2.line(image, (m, n), (m, n-d), (0,255,0), thickness=1, lineType=8, shift=0) #forehead
		cv2.circle(image , (u, i + z), 10, (0, 0, 255), -1)  #right_shoulder
		cv2.circle(image , (u + o, i + z), 10, (0, 255, 0), -1) #left_shoulder

	#Outputting
	cv2.imshow("Image", image)	
	if cv2.waitKey(0) & 0xFF == ord('q'):
		break

#Vacating and destroying all instances
cap.release()
cv2.destroyAllWindows()