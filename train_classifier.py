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
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from skimage.feature import hog

class Classifier(object):

    def __init__(self, vehicle_dir, non_vehicle_dir):
        self.vehicle_dir = vehicle_dir
        self.non_vehicle_dir = non_vehicle_dir
        self.total_images = 8792
        self.veh_features = []
        self.non_features = []
        self.X_scaler = None
        self.svc = None
        self.spa_pca = None
        self.hog_pca = None

    def create_image_lists(self):
        non_vehicle_filenames = glob.glob((non_vehicle_dir + '/*/*.png'), recursive=True)
        vehicle_filenames = glob.glob((vehicle_dir + '/*/*.png'), recursive=True)
        # Training classifier with equal number per catagory.
        print('Non:', len(non_vehicle_filenames))
        print('Veh:', len(vehicle_filenames))
        return non_vehicle_filenames[:self.total_images], vehicle_filenames[:self.total_images]

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

    def extract_all_features(self, image_list, is_veh):
        print('extracting features...')
        features = []
        spatial_features_stack = []
        hist_features_stack = []
        hog_features_stack = []

        for step, image_name in enumerate(image_list, 1):
            if (step % (self.total_images//10)) == 0:
                print(step)
            # Read image in as BGR scaled 0-255.
            image = cv2.imread(image_name)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            #image = mpimg.imread(image_name)
            spatial_feature_row = self.bin_spatial_features(image)
            if len(spatial_features_stack) == 0:
                spatial_features_stack = spatial_feature_row
            else:
                spatial_features_stack = np.concatenate((spatial_features_stack, spatial_feature_row), axis=0)

            hist_feature_row = self.color_hist_features(image)
            if len(hist_features_stack) == 0:
                hist_features_stack = hist_feature_row
            else:
                hist_features_stack = np.concatenate((hist_features_stack, hist_feature_row), axis=0)

            hog_feature_row = self.hog_features(image)
            if len(hog_features_stack) == 0:
                hog_features_stack = hog_feature_row
            else:
                hog_features_stack = np.concatenate((hog_features_stack, hog_feature_row), axis=0)

        spatial_features_stack = np.array(spatial_features_stack, dtype=np.float64)
        hist_features_stack = np.array(hist_features_stack, dtype=np.float64)
        hog_features_stack = np.array(hog_features_stack, dtype=np.float64)

        

        #total_features = np.concatenate((spatial_features_stack, hist_features_stack, hog_features_stack), axis=1)
        #total_features = np.concatenate((spatial_features_stack, hist_features_stack), axis=1)

        total_features = hog_features_stack
        print(total_features.shape)

        if is_veh == True:
            self.veh_features = total_features
        elif is_veh == False:
            self.non_features = total_features
        else:
            print('error in extract features')

    def normalize_features(self):
        print('normalizing features...')
        #print(self.non_features)
        # Create array stack of feature vectors.
        #X = np.vstack((self.veh_features, self.non_features)).astype(np.float64)
        X = np.concatenate((self.veh_features, self.non_features), axis=0)
        print(X.shape)


        # Fit a per-column scaler.
        for i in range(0, X.shape[0]):
            self.X_scaler = StandardScaler().partial_fit(X[i,0:].reshape(1, -1))
            if i % (X.shape[0]//10) == 0:
                print(i)
        # Apply the scaler to X.
        scaled_X = self.X_scaler.transform(X)
        return scaled_X

    def fit_classifier(self):
        print('fitting classifier...')
        y = np.hstack((np.ones((self.veh_features.shape[0])), np.zeros((self.non_features.shape[0]))))
        scaled_X = self.normalize_features()
        rand_state = np.random.randint(0, 100)
        X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2, random_state=rand_state)
        #self.svc = BaggingClassifier(LinearSVC(verbose=1, max_iter=10000), n_estimators=100, max_samples=2000, bootstrap=False)
        #self.svc = AdaBoostClassifier(LinearSVC(verbose=1, max_iter=10000), algorithm='SAMME')
        #self.svc = RandomForestClassifier(bootstrap=False)

        # Use a linear SVC (support vector classifier)

        self.svc = LinearSVC(verbose=1, max_iter=10000, C=1)

        # Train the SVC
        self.svc.fit(X_train, y_train)
        #print('Features used for this model = ', self.svc.n_features_)
        print('Test Accuracy of SVC = ', self.svc.score(X_test, y_test))
        #print('My SVC predicts: ', svc.predict(X_test[0:10].reshape(1, -1)))
        start = np.random.randint(0, len(y_test)-10)
        stop = start + 10
        print('My SVC predicts: ', self.svc.predict(X_test[start:stop]))
        print('For labels:      ', y_test[start:stop])

    def save_features_pickle(self):
        pickle.dump([self.veh_features, self.non_features], open('feature_pickle.p', 'wb'))

    def load_features_pickle(self):
        self.veh_features, self.non_features = pickle.load(open('feature_pickle.p', 'rb'))



start = time.time()
# Hardcoded image directories.
non_vehicle_dir = 'non-vehicles'
vehicle_dir = 'vehicles'

# Create classifier object.
Classifier = Classifier(non_vehicle_dir, vehicle_dir)

# This block used to extract features and save to pickle file.

non_veh, veh = Classifier.create_image_lists()
Classifier.extract_all_features(non_veh, False)
Classifier.extract_all_features(veh, True)
Classifier.save_features_pickle()


# This line used to load the features from pickle file.
#Classifier.load_features_pickle()

# Fit the classifier to extracted features.
Classifier.fit_classifier()
finish = time.time()
print('Total time to run', finish - start)
pickle.dump([Classifier.svc, Classifier.X_scaler], open('classifier_model.p', 'wb'))
print('Classifier pickled.')
