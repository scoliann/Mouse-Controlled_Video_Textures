import numpy as np
import cv2
import scipy.signal
from tqdm import tqdm
import time
import os
import glob
import errno
import shutil
import yaml
import hashlib
import json
import dill as pickle


def read_in_parameters():

	# This function reads in parameters from
	# a config YAML file
	with open('config.yaml', 'r') as cfg:
		config = yaml.load(cfg)
	return config


def setup_directories(video_file):

	# Define paths to each directory
	video_dir = video_file.split('.')[0]
	project_dir = os.path.join('output', video_dir)
	frames_dir = os.path.join(project_dir, 'frames')
	anchor_points_dir = os.path.join(project_dir, 'anchor_points')
	cache_dir = 'cache'

	# Create or reset specified directories
	if os.path.isdir(project_dir):
		shutil.rmtree(project_dir)
	if os.path.isdir(frames_dir):
		shutil.rmtree(frames_dir)
	if os.path.isdir(anchor_points_dir):
		shutil.rmtree(anchor_points_dir)
	if not os.path.isdir(cache_dir):
		os.makedirs(cache_dir)	
	os.makedirs(project_dir)
	os.makedirs(frames_dir)
	os.makedirs(anchor_points_dir)

	# Return directory names
	return video_dir, project_dir, frames_dir, anchor_points_dir, cache_dir


def show_frames(frames, show_wait_time_ms):

	# This function displays a video volume
	# to the user
	#
	# This function can often help with 
	# debugging
	#
	# Pressing Esc key will end visualization
	for f in frames:
		cv2.imshow('Frame Sequence Visualization', f)
		k = cv2.waitKey(show_wait_time_ms) & 0xff
		if k == 27:
			break
	cv2.destroyAllWindows()


def read_in_video(video_file, resize, frame_parse, total_frames, show, show_wait_time_ms):

	# Extract frames from video
	frames = []
	cap = cv2.VideoCapture(video_file)
	while True:
		ret, frame = cap.read()
		if ret:
			frames.append(frame)
		else:
			break
		if total_frames and (len(frames) == total_frames):
			break
	cap.release()

	# Preprocess frames
	#	- Keep every Nth frame
	#	- Resize frames
	frames_pp = []
	for index in range(len(frames)):
		frame = frames[index]
		if index % frame_parse == 0:
			frame = cv2.resize(frame, (0, 0), fx=resize, fy=resize)
			frames_pp.append(frame)

	# Construct video volume
	video_volume = np.array(frames_pp)

	# Show animation
	if show:
		show_frames(video_volume, show_wait_time_ms)

	# Return preprocessed frames
	return video_volume


def background_subtraction(frames_original, mb_kern, px_thresh, min_obj_size, show, show_wait_time_ms):

	def get_mask(fgmask):
		mask = np.dstack((fgmask, fgmask, fgmask))
		return mask

	# Perform background subtraction
	frames_bs = []
	bgs = cv2.createBackgroundSubtractorMOG2()
	for frame in frames_original:

		# Get a foreground mask
		fgmask = bgs.apply(frame)
		
		# Remove noise with median filter
		fgmask = cv2.medianBlur(fgmask, mb_kern)
	
		# Threshold remaining values to white or black
		_, fgmask = cv2.threshold(fgmask, px_thresh, 255, cv2.THRESH_BINARY)

		# Extract largest contour
		complete = False
		try:
			im, contours, hierarchy = cv2.findContours(fgmask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
			max_area, max_ctr = max([(cv2.contourArea(ctr), ctr) for ctr in contours])
			if max_area > min_obj_size:
				fgmask = np.zeros(fgmask.shape, fgmask.dtype)
				cv2.drawContours(fgmask, [ctr], 0, 255, -1)
				complete = True
		except:
			pass

		# Generate image without background
		if complete:
			frame[fgmask == 0] = (0, 0, 0)
			frames_bs.append(frame)

	# Construct video volume
	video_volume = np.array(frames_bs)

	# Show animation
	if show:
		show_frames(video_volume, show_wait_time_ms)

	# Return frames with backgrounds removed
	return video_volume


def get_anchor_points(video_volume):

	# This function calculate coordinates corresponding
	# to eight points around a rectangle, specifically:
	#
	#	- Top left
	#	- Top middle
	#	- Top right
	#	- Middle left
	#	- Middle right
	#	- Bottom left
	#	- Bottom middle
	#	- Bottom right
	#
	#	-----------------------------------
	#	|tl		tm		tr|
	#	|			          |
	#	|			          |
	#	|			          |
	#	|ml				mr|
	#	|			          |
	#	|			          |
	#	|			          |
	#	|bl		bm		br|
	#	-----------------------------------
	anchor_points = []
	dim_i = float(video_volume.shape[1])
	dim_j = float(video_volume.shape[2])
	anchors = ['tl', 'tm', 'tr', 'ml', 'mr', 'bl', 'bm', 'br']
	for anchor_code in anchors:
		if anchor_code == 'tl':
			coor = (0.0, 0.0)
		elif anchor_code == 'tm':
			coor = (0.0, dim_j / 2)
		elif anchor_code == 'tr':
			coor = (0.0, dim_j)
		elif anchor_code == 'ml':
			coor = (dim_i / 2, 0.0)
		elif anchor_code == 'mr':
			coor = (dim_i/2, dim_j)
		elif anchor_code == 'bl':
			coor = (dim_i, 0.0)
		elif anchor_code == 'bm':
			coor = (dim_i, dim_j / 2)
		elif anchor_code == 'br':
			coor = (dim_i, dim_j)
		anchor_points.append((anchor_code, coor))
	return anchor_points


def velocity_similarity_metrics(video_volume, anchor_points, velocity_window):

	'''
	Note:

	In VA and PVA, indicies (i, j) are mapped to frames (i, j)
	'''

	def calculate_velocities(video_volume, window):

		# Assume that velocity_window = 3
		# This function calculates the coordinates of
		#	the center of mass on frames i and i+3
		# Then, using the change in position of the 
		#	center of mass, an instantaneous
		#	velocity is calculated for frame i
		V = []
		for i in range(video_volume.shape[0] - window):

			# Calculate the center of mass of frame i
			frame_A_idx = i
			frame_A = video_volume[frame_A_idx,]
			A_cm_idxes = np.where(frame_A > 0.0)
			A_cm_i = np.mean(A_cm_idxes[0])
			A_cm_j = np.mean(A_cm_idxes[1])

			# Calculate the center of mass of frame
			# i + window
			frame_Z_idx = i + window
			frame_Z = video_volume[frame_Z_idx,]
			Z_cm_idxes = np.where(frame_Z > 0.0)
			Z_cm_i = np.mean(Z_cm_idxes[0])
			Z_cm_j = np.mean(Z_cm_idxes[1])

			# Calculate the velocity in the i and j
			# directions
			V_AZ_i = (Z_cm_i - A_cm_i) / (window)
			V_AZ_j = (Z_cm_j  - A_cm_j) / (window)

			# Store each velocity in a list
			# Note that the velocities stored at index 0
			#	in the list will correspond to frame 0, etc.
			V.append({'i': V_AZ_i, 'j': V_AZ_j})

		return V

	def calculate_angle(V_0, V_1):

		# This function calculates the angle, in degrees,
		# between two vectors
		numerator = np.dot(V_0, V_1)
		denominator = np.hypot(V_0[0], V_0[1]) * np.hypot(V_1[0], V_1[1])
		cos_angle = round(numerator / denominator, 8)
		angle = np.rad2deg(np.arccos(cos_angle))
		return angle

	def calculate_velocity_angles(V):

		# This function calculates the angle between the 
		# instantaneous velocities of every combination of
		# two frames (i, j) in the video volume
		VA = np.zeros((len(V), len(V)), dtype=np.float64)
		for i in range(VA.shape[0]):
			for j in range(VA.shape[1]):
				frame_i_Vi = V[i]['i']
				frame_i_Vj = V[i]['j']
				frame_j_Vi = V[j]['i']
				frame_j_Vj = V[j]['j']
				V_i = [frame_i_Vi, frame_i_Vj]
				V_j = [frame_j_Vi, frame_j_Vj]
				angle = calculate_angle(V_i, V_j)
				VA[i, j] = angle
		return VA

	def calculate_position_changes(video_volume, anchor_coor):

		# This function calculates the change in position
		# between the center of mass of each frame, and
		# the coordinates of a given anchor point
		PD = []
		for i in range(video_volume.shape[0]):

			# Calculate the center of mass for frame i
			frame_i = video_volume[i,]
			I_cm_idxes = np.where(frame_i > 0.0)
			I_cm_i = np.mean(I_cm_idxes[0])
			I_cm_j = np.mean(I_cm_idxes[1])

			# Calculate the change of position between
			# frame i and the anchor point
			PD_i = anchor_coor[0] - I_cm_i 
			PD_j = anchor_coor[1] - I_cm_j

			# Store each change of position in a list
			# Note that the change in position stored at index 0
			#	in the list will correspond to frame 0, etc.
			PD.append({'i': PD_i, 'j': PD_j})
		return PD

	def calculate_position_vs_velocity_angles(PD, V):

		# This function calculates the angles between the 
		# change of position and the instantaneous velocities
		# at each frame.
		# 
		# A small angle indicates that the center of mass is
		# moving in a direction similar to its instantaneous
		# velocity
		PVA = {}
		for anchor_code in PD:
			pos_deltas = PD[anchor_code]
			angles = []
			for i in range(len(V)):
				frame_i_Vi = V[i]['i']
				frame_i_Vj = V[i]['j']
				anchor_PDi = pos_deltas[i]['i']
				anchor_PDj = pos_deltas[i]['j']
				V_i = [frame_i_Vi, frame_i_Vj]
				V_j = [anchor_PDi, anchor_PDj]
				angle = calculate_angle(V_i, V_j)
				angles.append(angle)
			PVA[anchor_code] = angles
		return PVA

	# Calculate velocities at each frame
	V = calculate_velocities(video_volume, velocity_window)

	# Calculate angles between velocities
	VA = calculate_velocity_angles(V)

	# Calculate position changes between frames and anchor
	PD = {}
	for (anchor_code, anchor_coor) in anchor_points:
		PD[anchor_code] = calculate_position_changes(video_volume, anchor_coor)
		
	# Calculate the angle between position changes between 
	# frames and anchor, and velocity
	PVA = calculate_position_vs_velocity_angles(PD, V)

	# Shave off entries from video volume
	# Keeping frames only for which calculations
	#	could be performed
	video_volume = video_volume[:len(V),]

	# Return calculated results
	return video_volume, VA, PVA


def image_similarity_metric(video_volume, batch_size=10):

	'''
	Note:

	In D_0, indicies (i, j) are mapped to frames (i, j)
	'''

	def calculate_distance(A, B):
		return np.sum((A - B) ** 2, axis=(2, 3, 4)) ** 0.5

	# Initialize variables
	video_volume = video_volume.astype(np.float64)
	Row = video_volume[:, np.newaxis,]
	Col = video_volume[np.newaxis,]
	dimension = video_volume.shape[0]
	D_0 = np.zeros((dimension, dimension), dtype=video_volume.dtype)

	# Calculate distance in batches
	for idx_y in tqdm(range(int(np.ceil(D_0.shape[0] / float(batch_size))))):
		sl_y = slice(idx_y * batch_size, (idx_y + 1) * batch_size)
		for idx_x in range(int(np.ceil(D_0.shape[1] / float(batch_size)))):
			sl_x = slice(idx_x * batch_size, (idx_x + 1) * batch_size)
			R = Row[sl_y,]
			C = Col[:, sl_x]
			solved_segment = calculate_distance(R, C)
			D_0[sl_y, sl_x] = solved_segment

	# Return similarity matrix
	return D_0


def image_similarity_metric_with_dynamics(D_0, size=4):

	'''
	Note:

	In D_1_dym, indicies (i, j) are mapped to frames (i+1, j+2)
	'''

	# This function recalculates image similarity to
	# include dynamics
	#
	# This means that the similarity of two images is
	# recomputed to also consider the similarity of 
	# neighboring images
	#
	# This augmentation ensures that images with a 
	# high similarity not only look similar, but also
	# exhibit similar behavior (intra image motion, etc.)
	size = int(np.ceil(size / 2.0)) * 2
	row_increment = (size / 2) - 1
	col_increment = (size / 2)
	shape = (D_0.shape[0] - size + 1, D_0.shape[1] - size + 1)
	D_1_dym = np.zeros(shape, dtype=D_0.dtype)
	for i in range(D_1_dym.shape[0]):
		for j in range(D_1_dym.shape[1]):
			f_i = i + row_increment
			f_j = j + col_increment
			i_indicies = [f_i+index-row_increment for index in range(size)]
			j_indicies = [f_j+index-col_increment for index in range(size)]
			diff = 0.0
			for i_indx in range(len(i_indicies)):
				for j_indx in range(len(j_indicies)):
					k_i = i_indicies[i_indx]
					k_j = j_indicies[j_indx]
					weight = (size - np.absolute(i_indx - j_indx)) ** 2
					diff += (weight * D_0[k_i, k_j])
			D_1_dym[i, j] = diff
	return D_1_dym


def compound_similarity_metric(D_1_dym, VA, PVA, w_1, w_2, w_3):

	'''
	Note:

	In D_1,  indicies (i, j) are mapped to frames (i+1, j+2)

	Reminder:

	In VA and PVA, indicies (i, j) are mapped to frames (i, j)
	In D_1_dym, indicies (i, j) are mapped to frames (i+1, j+2)
	'''

	def normalize(A):
		return ((A - A.min()) / (A.max() - A.min())).astype(np.float64)

	# Normalize inputs
	D_1_dym = normalize(D_1_dym)
	VA = normalize(VA)
	PVA = normalize(np.array(PVA))

	# Update values in D_1_dym to consider velocity
	# similarity metrics
	D_1 = np.zeros(D_1_dym.shape, dtype=np.float64)
	for i in range(D_1.shape[0]):
		for j in range(D_1.shape[1]):
			i_idx = i + 1
			j_idx = j + 2
			score = (w_1 * D_1_dym[i, j]) + (w_2 * VA[i_idx, j_idx]) + (w_3 * PVA[j_idx])
			D_1[i, j] = score
	return D_1


def compound_similarity_metric_with_future_cost(D_1, p, a, sigma):

	'''
	Note:

	In D_2,  indicies (i, j) are mapped to frames (i+1, j+2)
	'''

	def apply_q_learning(D_1, p, a):

		# Apply Q Learning until convergence
		D_1_p = D_1.copy() ** p
		D_2 = D_1.copy() ** p
		old = D_2.copy() * -1
		iterator = 0
		while True:
			for i_idx in range(D_2.shape[0]):
				i = D_2.shape[0] - 1 - i_idx
				D_2[i,] = D_1_p[i,] + (a * np.min(D_2, axis=1))

			# End if converge
			if np.allclose(D_2, old, rtol=0.0, atol=0.5):
				break
			old = D_2.copy()

			# Print number of iterations
			iterator += 1
			print('Q-learning Iterations:\t{}'.format(iterator))

		# Return results
		return D_2

	# Apply Q-learning to update the calculated similarity
	# to anticipate (and hopefully avoid) points of great
	# future cost
	D_2_i = apply_q_learning(D_1, p, a)
	D_2_j = np.transpose(D_2_i)
	D_2 = (D_2_i + D_2_j) / 2.0
	return D_2


def transition_probability(D, sigma):

	# Calculate a scale factor equal to the average of 
	#	the elements in D divided by sigma
	# Create a new matrix P where P(i,j) = e^(-D_scaled(i,j))
	# Divide each element in P by the sum of its row
	#	This causes all rows to add up to 1.0
	scale_factor = (np.sum(D) / np.count_nonzero(D)) * sigma
	D_scaled = D / scale_factor
	P = np.exp(-1 * D_scaled)
	P_norm = P / P.sum(axis=1, keepdims=True)
	return P_norm


def apply_pruning(D_2, P_2, p_thresh):

	'''
	Note:

	In mask, prob_mask, prob_norm, and prob, indicies 
	(i, j) are mapped to frames (i+1, j+2)
	'''

	def check_ij(i, j, di, dj):
		if (0 <= i <= di-1) and (0 <= j <= dj-1):
			return True
		else:
			return False
	
	# Apply first step of pruning by only keeping
	# values that are local mins
	mask = np.zeros(D_2.shape, dtype=D_2.dtype)
	for i in range(mask.shape[0]):
		for j in range(mask.shape[1]):

			# Get neighbors
			neighbors = []
			if check_ij(i - 1, j, D_2.shape[0], D_2.shape[1]):
				neighbors.append(D_2[i - 1, j])
			if check_ij(i, j - 1, D_2.shape[0], D_2.shape[1]):
				neighbors.append(D_2[i, j - 1])
			if check_ij(i, j + 1, D_2.shape[0], D_2.shape[1]):
				neighbors.append(D_2[i, j + 1])
			if check_ij(i + 1, j, D_2.shape[0], D_2.shape[1]):
				neighbors.append(D_2[i + 1, j])

			# Check if min
			if (i != j + 1) and (D_2[i, j] < np.min(neighbors)):
				mask[i, j] = 1.0

	# Apply second step of pruning by:
	#	- Normalize values to sum to 1.0
	#		(Thereby making them interpretable
	#		 as probabilities)
	#	- Prune probabilities below threshold
	#	- Re-normalize remaining values to sum to 1.0
	prob = P_2 * mask
	for i in range(prob.shape[0]):
		total = np.sum(prob[i,])
		if total != 0.0:
			prob[i,] /= total
		else:
			prob[i,] = 0.0
	prob[prob < p_thresh] = 0.0
	for i in range(prob.shape[0]):
		total = np.sum(prob[i,])
		if total != 0.0:
			prob[i,] /= total
		else:
			prob[i,] = 0.0

	# Remove dead ends
	# In rows with no transitions, create a transition
	# probability of 1.0 from frame i to frame i+1
	for i in range(prob.shape[0]):
		row_prob_sum = np.sum(prob[i,])
		if row_prob_sum == 0.0:
			prob[i, i] = 1.0

	# Define prob mask
	prob_mask = np.zeros(prob.shape, dtype=prob.dtype)
	prob_mask[prob > 0.0] = 1.0

	# Normalize to pixel ranges
	prob_mask *= 255.0
	prob_norm = prob * 255.0

	# Return mask with local min, and probabilities
	return mask, prob_mask, prob_norm, prob


def array_to_img(arr):

	# Normalize values of an input array to fall between 0 and 255
	# so that the array can be saved as an image
	return (((arr - arr.min()) / (arr.max() - arr.min())) * 255).astype(np.uint8)


def get_cache_file_path(config, cache_dir):
	config_dict_str = json.dumps(config, sort_keys=True)
	config_hash = hashlib.md5(config_dict_str).hexdigest()
	cache_file_name = 'cache_{}.p'.format(config_hash)
	cache_file_path = os.path.join(cache_dir, cache_file_name)
	return cache_file_path


def video_texture_pre_computations(frames, project_dir, video_dir, anchor_points_dir, 
				   cache_dir, velocity_window, w_1, w_2, w_3, sigma, p, 
				   a, p_thresh, config):

	'''
	Note:

	In APP[key], indicies (i, j) are mapped to frames (i+1, j+2)
	'''
		
	# Perform steps to determine transition probabilities
	#	- Calculate the anchor points
	#	- Calculate velocity related similarity metrics
	#	- Calculate image similarity using Euclidean 
	#		distance between all frames (i, j)
	#	- Calculate image similarity with dynamics
	#		accounted for
	anchor_points = get_anchor_points(frames.copy())
	frames, VA, PVA = velocity_similarity_metrics(frames.copy(), 
						      anchor_points, 
						      velocity_window)
	D_0 = image_similarity_metric(frames.copy())
	D_1_dym = image_similarity_metric_with_dynamics(D_0.copy())

	# Perform calculations for directional influence
	# towards each anchor point
	APP = {}
	cache = {'save_visual_arg_list': [], 'post_computations': {}}
	for anchor_point in PVA:

		# Calculate similarity to include:
		#	- Image similarity
		#	- Dynamics
		#	- Velocity similarity metrics
		#		- Instantaneous velocity similarity
		#		- Position change and velocity similarity
		D_1 = compound_similarity_metric(D_1_dym.copy(), VA.copy(), PVA[anchor_point], w_1, w_2, w_3)

		# Augment similarity using Q-learning to help
		# avoid transitions with high future cost
		D_2 = compound_similarity_metric_with_future_cost(D_1.copy(), p, a, sigma)	

		# Calculate matrix of transition probabilities using
		# image similarity	
		P_0 = transition_probability(D_0.copy(), sigma)	

		# Calculate matrix of transition probabilities using
		# image similarity, dynamics, and velocity similarity
		# metrics		
		P_1 = transition_probability(D_1.copy(), sigma)	

		# Calculate matrix of transition probabilities using future
		# cost (via Q-learning) on top of a similarity metric that
		# incorporates: image similarity, dynamics, and velocity 
		# similarity metrics		
		P_2 = transition_probability(D_2.copy(), sigma)

		# Apply pruning via local mins and thresholding to calculate
		# a final probability transition matrix (i.e. Markov chain)
		#
		# Additionally, perform some additional calculations for 
		# purpose of visualization
		D_3_mask, P_3_mask, P_3_norm, prob = apply_pruning(D_2.copy(), P_2.copy(), p_thresh)	
		APP[anchor_point] = prob

		# Store all precomputations used in rendering visualizations
		# in a dictionary that will be cached
		save_dir = os.path.join(anchor_points_dir, anchor_point)
		cache['save_visual_arg_list'] += [[
							(D_0, 'D_0', anchor_point, save_dir, video_dir),
							(D_1, 'D_1', anchor_point, save_dir, video_dir),
							(D_2, 'D_2', anchor_point, save_dir, video_dir),
							(P_0, 'P_0', anchor_point, save_dir, video_dir),
							(P_1, 'P_1', anchor_point, save_dir, video_dir),
							(P_2, 'P_2', anchor_point, save_dir, video_dir),
							(D_3_mask, 'D_3_mask', anchor_point, save_dir, video_dir),
							(P_3_mask, 'P_3_mask', anchor_point, save_dir, video_dir),
							(P_3_norm, 'P_3_norm', anchor_point, save_dir, video_dir)
						 ]]

	# Store all remainin precomputations in a dictionary 
	# that will be cached
	cache['post_computations']['APP'] = APP
	cache['post_computations']['anchor_points'] = anchor_points
	cache['post_computations']['frames'] = frames

	# Cache useful computations
	cache_file_path = get_cache_file_path(config, cache_dir)
	pickle.dump(cache, open(cache_file_path, 'wb'))

	# Return transition probabilities
	return APP, anchor_points, cache


def get_video_texture_pre_computations(config, cache_dir, video_file, 
				       resize, frame_parse, num_frames, 
				       show, show_wait_time_ms, mb_kern, 
				       px_thresh, min_obj_size, project_dir, 
				       video_dir, anchor_points_dir, 
				       velocity_window, w_1, w_2, w_3, 
				       sigma, p, a, p_thresh):

	def save_visual(arr, name, anchor_point, save_dir, video_dir):
		img = array_to_img(arr)
		file_name = '{}_{}_{}.png'.format(video_dir, name, anchor_point)
		file_path = os.path.join(save_dir, file_name)
		cv2.imwrite(file_path, img)

	try:

		# Construct would-be name of cache file
		# And attempt to load cache
		cache_file_path = get_cache_file_path(config, cache_dir)
		cache = pickle.load(open(cache_file_path, 'rb'))

		# Extract all remaining precomputations from cache
		APP = cache['post_computations']['APP']
		anchor_points = cache['post_computations']['anchor_points']
		frames = cache['post_computations']['frames']

	except:

		# Perform video texture precomputations
		frames = read_in_video(video_file, resize, frame_parse, num_frames, show, show_wait_time_ms)
		frames = background_subtraction(frames, mb_kern, px_thresh, min_obj_size, show, show_wait_time_ms)
		APP, anchor_points, cache = video_texture_pre_computations(frames, project_dir, video_dir, 
									   anchor_points_dir, cache_dir, 
									   velocity_window, w_1, w_2, w_3, 
									   sigma, p, a, p_thresh, config)

	# Save visualization files for each anchor point
	for arg_list in cache['save_visual_arg_list']:
		save_dir = arg_list[0][3]
		if os.path.isdir(save_dir):
			shutil.rmtree(save_dir)
		os.makedirs(save_dir)
		for (arr, name, anchor_point, save_dir, video_dir) in arg_list:
			save_visual(arr, name, anchor_point, save_dir, video_dir)

	return APP, anchor_points, frames


mouse_i, mouse_j = -1, -1
break_var = False
def markov_synthesis(APP, anchor_points, frames, start_frame, show_wait_time_ms):

	def get_mouse_coor(event, x, y, flags, param):

		# Mouse callback function to get mouse coordinates
		global mouse_i, mouse_j, break_var
		if event == cv2.EVENT_LBUTTONDOWN:
			mouse_i, mouse_j = y, x
		elif event == cv2.EVENT_RBUTTONDOWN:
			break_var = True

	def find_nearest_anchor_point(anchor_points):

		# Find nearest anchor point
		global mouse_i, mouse_j
		anchor_dists = []
		for (anchor_code, coor) in anchor_points:
			distance = np.sqrt(((mouse_i - coor[0]) ** 2)+((mouse_j - coor[1]) ** 2))
			anchor_dists.append((distance, anchor_code))
		(distance, closest_anchor) = min(anchor_dists)
		return closest_anchor

	def find_next_frame(MC, current_frame):

		# Probabilistically determine the next frame
		if current_frame < MC.shape[0]:
			poss_next_frames_prob = MC[current_frame,]
			poss_next_frames = np.array([i for i in range(len(poss_next_frames_prob))]) + 1
			total_prob = np.sum(poss_next_frames_prob)
			try:
				next_frame = np.random.choice(poss_next_frames, p=poss_next_frames_prob)
				next_frame_prob = dict(zip(poss_next_frames, poss_next_frames_prob))[next_frame]
				if next_frame != current_frame + 1:
					did_jump = 'JUMPED'
				else:
					did_jump = ''
			except:
				next_frame = False
				next_frame_prob = 'Probability != 1.0, ---> {}'.format(total_prob)
				did_jump = ''
		else:
			next_frame = False
			next_frame_prob = 'At end'
			did_jump = ''
		return next_frame, next_frame_prob, did_jump

	# Run the video texture synthesis process
	#
	# Using user feedback, utilize the Markov chain
	# of the closest anchor point to determine
	# the next frame to transition to
	#
	# Because the computations underlying each
	# Markov chain value image similarity, instantaneous
	# velocity similarity, and position change vs
	# instantaneous velocity similarity, each
	# Markov chain will generate smooth transitions
	# of the video subject to its corresponding 
	# anchor point
	#
	# User feedback changes the selected anchor point
	# (and corresponding Markov chain), thereby allowing
	# the user to control the motion of the video subject
	synthesized_frames = []
	cv2.namedWindow('Markov Synthesis')
	cv2.setMouseCallback('Markov Synthesis', get_mouse_coor)
	current_frame_idx = start_frame
	global break_var
	while True:

		# Find nearest anchor point
		anchor_code = find_nearest_anchor_point(anchor_points)

		# Get the associated Markov chain
		MC = APP[anchor_code]

		# Choose a next frame from the probability matrix
		next_frame_idx, next_frame_prob, next_frame_did_jump = find_next_frame(MC, current_frame_idx)

		# If an error occurs, print message and return
		if not next_frame_idx:
			print('Error:\t{}'.format(next_frame_prob))
			return synthesized_frames
		else:
			synthesized_frames.append((next_frame_idx, next_frame_prob, next_frame_did_jump))

		# Display results
		cv2.imshow('Markov Synthesis', frames[next_frame_idx])
		k = cv2.waitKey(show_wait_time_ms) & 0xff
		if k == 27:
			break

		# Set variables
		current_frame_idx = next_frame_idx

		# Break if detected
		if break_var:
			break		
	cv2.destroyAllWindows()

	# Return results
	return synthesized_frames


def save_video_frames(frame_sequence, frames, frames_dir):

	# Save original video texture
	video_texture_frames = []
	order_counter = 0
	for (f_idx, f_prb, f_jmp) in frame_sequence:

		# Save frames
		frame = frames[f_idx,]
		file_name = '{}_frame{}_{}.png'.format(order_counter, f_idx, f_jmp)
		file_path = os.path.join(frames_dir, file_name)
		cv2.imwrite(file_path, frame)
		video_texture_frames.append(frame)
		order_counter += 1	

		# Print results to terminal
		print('')
		print('Frame Progression:\t{}\t{}\t{}'.format(f_idx, f_prb, f_jmp))

	# Set data types for video volumes
	video_texture_frames = np.array(video_texture_frames).astype(np.float64)
	video_texture_frames = [array_to_img(f) for f in video_texture_frames]

	# Return frames of final video textures
	return video_texture_frames


def make_video(frames, file_name_segment, project_dir, video_dir, fps):

	# Save video texture frames as a video
	file_name = '{}_{}.avi'.format(video_dir, file_name_segment)
	save_path = os.path.join(project_dir, file_name)
	fourcc = cv2.VideoWriter_fourcc(*'DIVX')
	shape = (frames[0].shape[1], frames[0].shape[0])
	out = cv2.VideoWriter(save_path, fourcc, fps, shape)
	for frame in frames:
		out.write(frame)
	out.release()
	cv2.destroyAllWindows()	


def main():

	# Read parameters from config file
	config = read_in_parameters()
	video_file = config['video_file']
	resize = config['resize']
	frame_parse = config['frame_parse']
	mb_kern = config['mb_kern']
	px_thresh = config['px_thresh']
	min_obj_size = config['min_obj_size']
	velocity_window = config['velocity_window']
	w_1 = config['w_1']
	w_2 = config['w_2']
	w_3 = config['w_3']
	sigma = config['sigma']
	p = config['p']
	a = config['a']
	p_thresh = config['p_thresh']
	start_frame = config['start_frame']
	num_frames = config['num_frames']
	fps = config['fps']
	show = config['show']
	show_wait_time_ms = config['show_wait_time_ms']

	# Do initial processing
	(video_dir, project_dir, frames_dir, 
	 anchor_points_dir, cache_dir) = setup_directories(video_file)

	# Do required preprocessing
	# Or read results of preprocessing from cache
	APP, anchor_points, frames = get_video_texture_pre_computations(config, cache_dir, video_file, 
									resize, frame_parse, num_frames, 
									show, show_wait_time_ms, mb_kern, 
									px_thresh, min_obj_size, project_dir, 
									video_dir, anchor_points_dir, 
									velocity_window, w_1, w_2, w_3, 
									sigma, p, a, p_thresh)

	# Construct video with mouse input
	#	- Left click	- directs video subject towards mouse
	#	- Right click	- ends synthesis process
	frame_sequence = markov_synthesis(APP, anchor_points, frames, start_frame, show_wait_time_ms)

	# Save frames
	video_texture_frames = save_video_frames(frame_sequence, frames, frames_dir)

	# Play frames
	show_frames(video_texture_frames, show_wait_time_ms)

	# Create video and save
	make_video(video_texture_frames, 'texture', project_dir, video_dir, fps)

	# Save original input video in project directory
	shutil.copy2(video_file, project_dir)


if __name__ == '__main__':
	main()



















