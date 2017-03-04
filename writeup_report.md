##Writeup Template
###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[hog_car1]: ./hog_car1.png
[hog_car2]: ./hog_car2.png
[hog_noncar1]: ./hog_noncar1.png
[hog_noncar2]: ./hog_noncar2.png
[cars_examples]: ./cars_examples.png
[noncars_examples]: ./noncars_examples.png


[scv1]: ./svc1.png
[scv2]: ./svc2.png

[cnn1]: ./cnn1.png
[cnn2]: ./cnn2.png

[heat1]: ./heat1.png
[heat2]: ./heat2.png
[heat3]: ./heat3.png
[heat4]: ./heat4.png



[label1]: ./label1.png
[label2]: ./label2.png
[label3]: ./label3.png
[label4]: ./label4.png



[drawn1]: ./drawn1.png


[project_video_output]: ./project_video_output.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the first code cell of the IPython notebook Vehicle-Detection.ipynb.

Here are some examples of `vehicle` classes:

![alt text][cars_examples]

and `non-vehicle` class:

![alt text][noncars_examples]


I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `RGB` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

Hog features of `vehicle` classes:

![alt text][hog_car1]

![alt text][hog_car2]

Hog features of `non-vehicle` classes:


![alt text][hog_noncar1]

![alt text][hog_noncar2]


####2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and found that a combination of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)` worked well for my classifier.

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

The code for my classifier training is located in the 7th code cell of the IPython notebook Vehicle-Detection.ipynb.

First I tried a Linear classifier and combined HOG features with color histogram and spatial binning, but I got a very low accuracy around 0.5. Then I decided to try the NuSVC classifier, used only HOG features and got a higher accuracy of 0.9163. I finally settled for this classifier.

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I used the hog sub-sampling window search discussed in the lesson and implemented it in the 13th cell of the IPython notebook Vehicle-Detection.ipynb. In my implementation, I tried different parameters and those that worked well for me are: window = 64, pixels per cell = 8, step = 2 (one step spans over 2 cells).


####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

I tried different parameters in my sliding window search implementation to optimize the results, notably to reduce false positives and detect actual cars present in the image.

Here are some example images genenated by my pipeline:

![alt text][scv1]
![alt text][scv2]

Besides the NuSVC classifier, I reused my neural network model in the Behavioral Cloning project (the code is in the IPython notebook Keras-Vehicle-Classifier.ipynb.) to train another model. It proved to work better than my NuSVC classifier. For the output video, I used my neural network model in my pipeline. Here are some example images generated by the use of my neural network model:

![alt text][cnn1]
![alt text][cnn2]

---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video_output.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

The code for my pipeline is in the 16th cell of the IPython notebook Vehicle-Detection.ipynb. I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle. I constructed bounding boxes to cover the area of each blob detected. 

When going from frame to frame, I used detections in previous frames as indications of possible vehicle positions in the new frame.

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are four frames and their corresponding heatmaps:

![alt text][heat1]
![alt text][heat2]
![alt text][heat3]
![alt text][heat4]


### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:

![alt text][label1]
![alt text][label2]
![alt text][label3]
![alt text][label4]

### Here the resulting bounding boxes are drawn onto the last frame in the series:

![alt text][drawn1]



---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I find that a linear classifier does not provide a high accuracy. I tried different scenarios in which I used only combine color histogram and spatial binning or HOG features, or all of them. However, I didn't get an accuracy higher than 0.6 with a linear classifier.

Then I tried NuSVC and it worked better for me even if I used only HOG features to train it.

It is also important to have a good training data that has a balanced numbers of car and non-car images.

Beside accuracy, vehicle tracking in real time might require a very high performance algorithm. But for me, it took several second to proccess a frame. So I think, performance is one important issue that needs further improvement.
