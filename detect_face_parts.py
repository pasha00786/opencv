# Written by Ashutosh Gupta
# Pyhton code to generate a funky frame over the human face using OpenCV and dlib library.

from imutils import face_utils
import numpy as np
import imutils
import dlib
import cv2
from utilities import image_resize
import matplotlib.pyplot as plt

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat') #Loading the facial HOG Detector

#cap = cv2.VideoCapture(0) #Getting frames from webcam.

#Instantiating facial index
part_1 = 'right_eyebrow'
part_2 = 'left_eyebrow'
part_3 = 'nose'
part_4 = 'jaw'

#Loding Filters
t_filter = cv2.imread("t.png", -1)
g_filter = cv2.imread("g.png", -1)


while 1:  #to continously stream the frames
	#_, image = cap.read()
	image = cv2.imread('ashu.jpg')
	image = imutils.resize(image, width=500)
	#Adding alpha channel
	image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
	gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY) #Grayscaled image to have fast and efficient detection
	rects = detector(gray, 1)
	# print(gray.shape)
	Y, X = gray.shape
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

		# for (x, y) in shape[k:e]:
		# 	cv2.circle(image, (x, y), 1, (0, 0, 255), -1)

		#Getting bounding boxes coords
		(a, b, c, d) = cv2.boundingRect(np.array([shape[y:z]])) #nose coords
		(x, y, w, h) = cv2.boundingRect(np.array([shape[i:j]])) #r-eyebrow coords
		(p, q, r, s) = cv2.boundingRect(np.array([shape[l:m]])) #l-eyebrow coords
		(u, i, o, z) = cv2.boundingRect(np.array([shape[k:e]])) #jaw-line coords

		#Calculating the originating coords and height of the forehead ROI 
		m, n = int((x+w+p)*0.5) , int((y+q)*0.5)
		d = int(d*0.75)

		#Shoulder Coords
		rs_x, rs_y = u, i+z
		ls_x, ls_y = u + o, i+z
		cv2.circle(image , (rs_x, rs_y), 5, (0, 255, 255), -1)  #right_shoulder
		#cv2.circle(image , (ls_x, ls_y), 5, (0, 255, 0), -1) #left_shoulder
		cv2.circle(image , (Y, X), 5, (0, 255, 0), -1)

		# T-filter
		tx = x+w
		ty = q - d
		tw = p-(x+w)
		th = d
		roi_t = gray[ty: ty + th, tx: tx + tw]
		T_ = image_resize(t_filter.copy(), height = th)
		Tw, Th, Tc = T_.shape
		for i in range(0, Tw):
			for j in range(0, Th):
				if T_[i, j][3] != 0: # alpha 0
						image[ty + i, tx + j] = T_[i, j]

		# G-filter

		#Left
		glx = ls_x - int(0.12*ls_x)
		gly = ls_y
		glw = w + int(0.12*ls_x)
		glh = Y - ls_y
		
		roi_g = gray[gly: gly + glh, glx: glx + glw]
		G_ = image_resize(g_filter.copy(), width = glw, height = glh)
		Gw, Gh, Gc = G_.shape
		for i in range(0, Gw):
			for j in range(0, Gh):
				if G_[i, j][3] != 0: # alpha 0
						image[gly + i, glx + j] = G_[i, j]


		#right

		#cv2.rectangle(image, (glx, gly), (glx + glw, gly + glh), (255, 255, 0), 1)
		#cv2.rectangle(image, (rs_x + int(0.12*rs_x), rs_y), (rs_x - int(1.3*w), Y), (255, 255, 0), 1)
		#Overlapping ROI over the frame
		#cv2.rectangle(image, (tx, ty), (tx+tw, ty-th), (255, 255, 0), 1)
		#cv2.line(image, (m, n), (m, n-d), (0,255,0), thickness=1, lineType=8, shift=0) #forehead
		
		# cv2.rectangle(image, (u, i), (u+o, i+z), (255, 255, 0), 1)

	image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)	
	cv2.imshow('img',image)
	#Outputting	
	if cv2.waitKey(0) & 0xFF == ord('q'):
		break

#Vacating and destroying all instances
# cap.release()
cv2.destroyAllWindows()