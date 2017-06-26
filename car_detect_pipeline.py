import os
import numpy as np
import argparse
import cv2
import time
import glob
import pickle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from skimage.feature import hog
from scipy.ndimage.measurements import label
from moviepy.editor import *

class Frame(object):
	def __init__(self, X_scaler, svc):
		
		self.X_scaler = X_scaler
		self.svc = svc
		self.frames_to_ave = 20
		self.heatmap_threashold = 12
		self.master_heatlist = []

	def search_frame(self, x_start_stop=[600,None], y_start_stop=[None,None], xy_size=(64,64), xy_overlap=(0.25,0.25)):
		# If no start-stop values given, use whole image.
		if x_start_stop[0] == None:
			x_start_stop[0] = 0
		if x_start_stop[1] == None:
			x_start_stop[1] = self.img.shape[1]
		if y_start_stop[0] == None:
			y_start_stop[0] = 0
		if y_start_stop[1] == None:
			y_start_stop[1] = self.img.shape[0]

		xspan = x_start_stop[1] - x_start_stop[0]
		yspan = y_start_stop[1] - y_start_stop[0]

		x_pix_per_step = np.int(xy_size[0]*(1 - xy_overlap[0]))
		y_pix_per_step = np.int(xy_size[1]*(1 - xy_overlap[1]))

		nx_buffer = np.int(xy_size[0]*(xy_overlap[0]))
		ny_buffer = np.int(xy_size[1]*(xy_overlap[1]))
		nx_windows = np.int((xspan-nx_buffer)/x_pix_per_step)
		ny_windows = np.int((yspan-ny_buffer)/y_pix_per_step)

		window_list = []

		for ys in range(ny_windows):
			for xs in range(nx_windows):
				# Calc window position.
				startx = xs*x_pix_per_step + x_start_stop[0]
				endx = startx + xy_size[0]

				starty = ys*y_pix_per_step + y_start_stop[0]
				endy = starty + xy_size[1]

				window_list.append(((startx, starty), (endx, endy)))

		return window_list

	def predict_boxes(self, window_list):
		hot_boxes = []
		for points in window_list:
			startx, starty = points[0]
			endx, endy = points[1]
			img_sample = self.img[starty:endy, startx:endx, 0:]
			# Resize img down to same size training was done with.
			img_sample = cv2.resize(img_sample, (64, 64))

			# Find features of box.
			spatial_features = self.bin_spatial_features(img_sample)
			spatial_features = np.array(spatial_features, dtype=np.float64)
			#spatial_features = spa_pca.transform(spatial_features)

			hist_features = self.color_hist_features(img_sample)
			hist_features = np.array(hist_features, dtype=np.float64)

			hog_features = self.hog_features(img_sample)
			hog_features = np.array(hog_features, dtype=np.float64)
			#hog_features = hog_pca.transform(hog_features)
			
			# Concatenate features to vector.
			#total_features = np.concatenate((spatial_features, hist_features, hog_features), axis=1)
			# Normalize features using scaler from training.
			total_features = hog_features
			total_features = self.normalize_features(total_features)

			if int(self.svc.predict(total_features.reshape(1,-1))) == 1:
				hot_boxes.append(((startx, starty), (endx, endy)))
				#print(self.svc.predict(total_features.reshape(1,-1)))
		return hot_boxes

	def draw_boxes(self, bboxes, color=(0, 0, 255), thick=4):
		# Make a copy of the image
		imcopy = np.copy(self.img)
		# Iterate through the bounding boxes
		for bbox in bboxes:
			# Draw a rectangle given bbox coordinates
			size = bbox[1][0] - bbox[0][0]
			if size <= 100:
				color = (255,0,0)
			elif size < 200:
				color = (0,255,0)
			else:
				color = (0,0,255)
			cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
		# Return the image copy with boxes drawn
		return imcopy

	def bin_spatial_features(self, img, size=(32, 32)):
		# Lower resolution and flatten to vector.
		features = cv2.resize(img, size).ravel().reshape(1,-1)
		return features

	def color_hist_features(self, img, nbins=50, bins_range=(0, 256)):
		channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
		channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
		channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
		# Concatenate the histograms into a single feature vector
		hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
		# Return the individual histograms, bin_centers and feature vector
		return hist_features.reshape(1,-1)

	def hog_features(self, img, orient=11, pix_per_cell=8, cell_per_block=2, vis=False, feature_vec=True):
		img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
		if vis == True:
			# Use skimage.hog() to get both features and a visualization
			features, hog_image = hog(img, orientations=orient,
			pixels_per_cell=(pix_per_cell, pix_per_cell), 
			cells_per_block=(cell_per_block, cell_per_block),
			transform_sqrt=False,
			visualise=True, feature_vector=False)
			return features, hog_image
		else:      
			# Use skimage.hog() to get features only
			features = hog(img, orientations=orient, 
			pixels_per_cell=(pix_per_cell, pix_per_cell), 
			cells_per_block=(cell_per_block, cell_per_block),
			transform_sqrt=False,
			visualise=False, feature_vector=feature_vec)
			return features.reshape(1,-1)

	def normalize_features(self, array):
		# Create array stack of feature vectors.
		array = np.array(array, dtype=np.float64)

		# Scale the array using scaler loaded from pickle file.
		scaled_array = self.X_scaler.transform(array.reshape(1,-1))
		return scaled_array

	def add_heat(self, heatmap, bboxes):
		# Iterate through list of boxes.
		for box in bboxes:
			# Add += 1 for each pixel in each box to the heatmap.
			heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
		return heatmap

	def apply_threshold(self, heatmap, threshold):
		# Zero out pixels below threshold.
		heatmap[heatmap <= threshold] = 0
		return heatmap

	def draw_labeled_bboxes(self, img, labels):
		# Iterate through all detected cars
		for car_number in range(1, labels[1]+1):
			# Find pixels with each car_number label value
			nonzero = (labels[0] == car_number).nonzero()
			# Identify x and y values of those pixels
			nonzeroy = np.array(nonzero[0])
			nonzerox = np.array(nonzero[1])
			# Define a bounding box based on min/max x and y
			bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
			# Draw the box on the image
			cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 4)
			# Return the image
		return img

	def add_to_master_heat(self, heatmap):
		
		self.master_heatlist.append(heatmap)
		
		if len(self.master_heatlist) > self.frames_to_ave:
			del self.master_heatlist[0]
			# If each list element is a heatmap, add them together.
			total_heatmap = np.zeros((self.img.shape[0], self.img.shape[1]))
			for elem in self.master_heatlist:
				total_heatmap = np.add(elem, total_heatmap)
			return total_heatmap
		else:
			return heatmap

	def update_frame(self, img):
		self.img = img 

		# Specify the size and start-stop of search boxes.
		#........................................y is vertical
		search_boxes = single_frame.search_frame(x_start_stop=[750,None], y_start_stop=[400,550], xy_size=(100,100))
		search_boxes += single_frame.search_frame(x_start_stop=[780,None], y_start_stop=[400,600], xy_size=(150,150))
		search_boxes += single_frame.search_frame(x_start_stop=[920,None], y_start_stop=[450,None], xy_size=(200,200))

		# Predict whether there's a car in each search box.
		hot_boxes = single_frame.predict_boxes(search_boxes)

		# Test image of hot_boxes overlayed on image.
		drawn_hot_boxes = single_frame.draw_boxes(hot_boxes)

		# Create blank heatmap.
		heatmap = np.zeros((self.img.shape[0],self.img.shape[1]))

		# Add the hot boxes from the predictor to the heatmap.
		heatmap = single_frame.add_heat(heatmap, hot_boxes)

		master_heatmap = single_frame.add_to_master_heat(heatmap)

		culled_heatmap = single_frame.apply_threshold(master_heatmap, self.heatmap_threashold)

		# Sidebar output.
		d = drawn_hot_boxes
		c = cv2.merge((culled_heatmap, culled_heatmap, culled_heatmap))
		m = cv2.merge((master_heatmap, master_heatmap, master_heatmap))
		#print(d.shape)
		#print(c.shape)
		#print(m.shape)
		sidebar = np.concatenate(( d, m, c), axis=0)
		sidebar = cv2.resize(sidebar, (427,720))

		#heatmap = single_frame.apply_threshold(heatmap, self.heatmap_threashold)

		# Create the labeled array from the heatmap.
		labels = label(culled_heatmap)

		# Draw boxes around the labeled parts of the labeled heatmap.
		overlayed_image = single_frame.draw_labeled_bboxes(self.img, labels)

		#print('video', overlayed_image.shape)
		#print('sidebar', sidebar.shape)

		overlayed_image = np.concatenate((overlayed_image, sidebar), axis=1)


		#return overlayed_image
		return overlayed_image




# Open pickle file and load svc and scaler objects.
svc, X_scaler = pickle.load(open('classifier_model.p', 'rb'))

# Grab filename from terminal.
parser = argparse.ArgumentParser(description='Get arg filename.')
parser.add_argument('file')
args = parser.parse_args()
arg_filename = args.file

# Create frame object.
single_frame = Frame(X_scaler, svc)



'''
# This block used to ouput window search and prediction images for writeup.
test_frame_filename = 'test_images/test1.jpg'
test_image = mpimg.imread(test_frame_filename)
single_frame.img = test_image
search_boxes = single_frame.search_frame(x_start_stop=[750,None], y_start_stop=[400,550], xy_size=(100,100))
search_boxes += single_frame.search_frame(x_start_stop=[780,None], y_start_stop=[400,600], xy_size=(150,150))
search_boxes += single_frame.search_frame(x_start_stop=[920,None], y_start_stop=[450,None], xy_size=(200,200))
search_windows = single_frame.draw_boxes(single_frame.predict_boxes(search_boxes))
plt.imshow(search_windows)
plt.show()
'''


'''
# This block used to output images for writeup
input_clip = VideoFileClip(arg_filename)
input_clip = input_clip.subclip(38,40)
frame_list = list(input_clip.iter_frames())

heatmap_list = []
for i in frame_list[:6]:
	image = single_frame.update_frame(i)

#heatmaps = np.concatenate((heatmap_list), axis=1)
#heatmaps = cv2.merge((heatmaps, heatmaps, heatmaps))

#images = np.concatenate((frame_list[:6]), axis=1)

plt.imshow(image)
plt.show()
'''



# This block for final video stream.
input_clip = VideoFileClip(arg_filename)
output_filename = 'output_' + arg_filename
# Only read first three seconds for testing.
input_clip = input_clip.subclip(20,45)
new_clip = input_clip.fl_image(single_frame.update_frame)
new_clip.write_videofile(output_filename, audio=False)
