
import cv2
import numpy as np


MIN_MATCHES = 50
detector = cv2.ORB_create(nfeatures=5000)

index_params = dict(algorithm = 1, trees=3)
search_params = dict(checks=100)
flann = cv2.FlannBasedMatcher(index_params,search_params)



def load_input():
	input_image = cv2.imread('mouse_pad.jpg')
	input_image = cv2.resize(input_image, (400,550),interpolation=cv2.INTER_AREA)
	gray_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

	keypoints, descriptors = detector.detectAndCompute(gray_image, None)

	return gray_image,keypoints, descriptors



def compute_matches(descriptors_input, descriptors_output):
	
	if(len(descriptors_output)!=0 and len(descriptors_input)!=0):
		matches = flann.knnMatch(np.asarray(descriptors_input,np.float32),np.asarray(descriptors_output,np.float32),k=2)
		good = []
		for m,n in matches:
			if m.distance < 0.68*n.distance:
				good.append([m])
		return good
	else:
		return None




if __name__=='__main__':

	input_image, input_keypoints, input_descriptors = load_input()

	cap = cv2.VideoCapture(0)
	ret, frame = cap.read()

	while(ret):
		ret, frame = cap.read()

		if(len(input_keypoints)<MIN_MATCHES):
			continue
		frame = cv2.resize(frame, (700,600))
		frame_bw = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		output_keypoints, output_descriptors = detector.detectAndCompute(frame_bw, None)
		matches = compute_matches(input_descriptors, output_descriptors)

		if(matches!=None):
			output_final = cv2.drawMatchesKnn(input_image,input_keypoints,frame,output_keypoints,matches,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
			cv2.imshow('Final Output', output_final)
		else:
			cv2.imshow('Final Output', frame)
		key = cv2.waitKey(5)
		if(key==27):
			break