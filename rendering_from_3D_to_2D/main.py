import numpy as np
import cv2 as cv
import scipy.io as io

def distort_correction(coor,dis_para):
	for i in range(len(coor[0])):
		r_square = coor[0][i]**2 + coor[1][i]**2
		correction = 1 + dis_para[0]*r_square + dis_para[1]*(r_square ** 2) + dis_para[4]*r_square**3
		coor[0][i] = coor[0][i]*correction
		coor[1][i] = coor[1][i]*correction
	return coor

def project_points(X, K, R, T, distortion_flag=False,distortion_params=None):

	# create homogeneous coordinates
	homo_X = np.transpose(np.array([[X[0][i],X[1][i],X[2][i],1] for i in range(len(X[0]))]))
	#concatenate R and T
	RT = np.concatenate((R, T), axis=1)
	#RT x homo
	tmp = np.dot(RT, homo_X)
	#scaling, dividing by z coordinates
	scaled = np.transpose(np.array([[tmp[0][i]/tmp[2][i], tmp[1][i]/tmp[2][i], tmp[2][i]/tmp[2][i]] for i in range(len(tmp[0]))]))

	if distortion_flag:
		scaled = distort_correction(scaled,distortion_params)
	# K x (x,y,z) & get only x and y coordinates
	X_camera = np.dot(K,scaled)
	return X_camera

def project_and_draw(img, X_3d, K, R, T, distortion_flag=False , distortion_parameters=None):

	#call a function and get 2D coordinates
	coor_2d = project_points(X_3d,K,R,T,distortion_flag,distortion_parameters)

	#change color
	if distortion_flag:
		color_tuple = (0, 255, 0) #green
	else:
		color_tuple = (0, 0, 255)  # red

	#plotting
	for i in range(len(coor_2d[0])):
		cv.circle(img, (int(coor_2d[0][i]),int(coor_2d[1][i])), 2, color_tuple, thickness=3)

	#show image and save as "output.jpg"
	cv.imshow("foo", img)
	cv.waitKey()
	cv.imwrite("output.jpg", img)

	return True

if __name__ == '__main__':
	base_folder = './data/'

	image_num = 20
	data = io.loadmat('./data/ex1.mat')
	X_3D = data['X_3D'][0]
	TVecs = data['TVecs']		# Translation vector: as the world origin is seen from the camera coordinates
	RMats = data['RMats']		# Rotation matrices: converts coordinates from world to camera
	kc = data['dist_params']	# Distortion parameters (k1,k2,p1,p2,k5)
	Kintr = data['intinsic_matrix']	# K matrix of the cameras
	print kc
	imgs = [cv.imread(base_folder+str(i).zfill(5)+'.jpg') for i in range(TVecs.shape[0])]
	project_and_draw(imgs[image_num],
					 X_3D,
					 Kintr,
					 RMats[image_num],
					 TVecs[image_num],
					 distortion_flag=True,
					 distortion_parameters=kc)
