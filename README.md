# CarND-VehicleDetection-P5

**Vehicle Detection Project**


[//]: # (Image References)
[image1]: ./writeup_pics/cars_not_cars.png
[image2]: ./writeup_pics/gray_HOG.png
[image3]: ./writeup_pics/search_windows.png
[image4]: ./writeup_pics/predicted_search_windows.png
[image5]: ./writeup_pics/six_frames.png
[image6]: ./writeup_pics/six_heatmaps.png
[image7]: ./writeup_pics/six_labels.png
[image8]: ./writeup_pics/six_frame_output.png

[video1]: ./output_project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points 

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.

Below I will explain how I implemented each part of the project as required by the rubric.

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

In the file 'train_classifier.py' under the 'Classifier' class there is a method called 'hog_features()' starting on line 53. To exctract the HOG features from the training images first two lists are generated. One list of vehicle filenames and one list of non-vehicle filenames. Once each of the lists are generated they are sent through the 'hog_features' method which uses the 'skimage.hog()' function. The hog features are returned as a single vector of results for training.

Here is an example of some images from of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  In the end I used a simple pipline of converting the image to grayscale before passing it to the 'hog_features()' function.

Here is an example using the `GRAY` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various parameters to balance distinct features with minimal memory taxation. Increasing the parameters too much taxed my system and caused memory errors. PCA was attemted to reduce the amount of features but the accuracy of the classifier was effected too much. In the end I simply increased the orientation to 11, used a pixel per cell of 8 and a cell per block of 2. This seemed to give enough detail why still allowing my computer to train the model. 

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

The first thing I did in order to save time was extract all the features I wanted and save them to a pickle file. This allowed me to do the feature extraction once, which took some time, and experiment with training the algorithm later. At line 141 in 'train_classifier.py' I have the 'fit_classifier()' fuction that trains the LinearSVM classifier using the 'train_test_split()' function. This allows some of the testing data to be left out of training to be used as a benchmark for the classifier.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

At line 25 of 'car_detect_pipeline.py' you'll see the 'search_frame()' fuciton. Here I started with the code from the sliding windows lesson. For my initial parameters I used a photo editing program to open the test images and measure some windows that may find a car in the scene. With these parameters I then went on to my next approach.

To start tuning my search windows I added a function that would generate an extra video stream showing all of the positively predicted search windows, 'hot_boxes', in real time. I concatenated this video along with two different heatmaps to the right of the main video. I overlayed the 'hot_boxes' on the video in colors according to their shapes. This way I could experiment with three shapes, red, green and blue. This allowed me to watch which windows were doing the work and which were lacking. It also allowed me to pinpoint sections of the video that were causing problems.

Below is an example of my sliding window search from when I was experimenting with size and location of the windows. In this experimentation I also color coded the drawn boxes.

![alt text][image3]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

At first I started with HOG features, spatially binned features as well as the color histograms. I thought more features would make the classifier work well but in the end I only used the HOG features. Using the other groups caused too many false positives. It was also very memory intensive and I couldn't test as many classifier algorithms as I wanted. In the end HOG with some frame averaging seems to work well. Below is the same frame image as above, only with the prediction classifier run over the sliding window search. As you can see it picks up the cars fairly well. This is before any averaging or heatmaps are used.

![alt text][image4]
---

### Video Implementation

#### 1. Provide a link to your final video output.
Here's a [link to my video result][video1]

#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions. This thresholded heatmap was then added to a 'master_heatmap' that kept the running average over 'n' frames. This allowed me to smooth the data a bit. I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]
![alt text][image6]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image7]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image8]



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The major problems I had in this project were the memory restrictions on my low-powered laptop. If I were to start over I would have used AWS or a more powerful computer. Once I had the pipeline built it was hard to port it over to another system. The biggest implementaion problem of this project I faced was detecting false positives in the shadows on the roadway. To make it more robust I would collect images of shadows like in the video and add them to my training set. I have a feeling the classifier was more detecting 'lots of curved lines' vs 'not many curved lines' in each search window.  

