
## Vehicle Detection Project

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* a color transform and append binned color features to my HOG feature vector.　features　normalization and randomization of training and testing dataset are conducted
* a sliding-window technique is adopted to propose candidate regions for car, and the regions are input into a trained classifier
* a pipeline works on a video stream and a heat map is created for recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

### Submissiones

* CarND-Vehicle-Dection.ipynb is the project solution , all the code contains in it.
* svc_pickle.p is the model and feature parameters
* README.md it the writeup file of the project
* test_result.mp4 is the performance on the test_video.mp4
* result.mp4 is the performance on the project_video.mp4
---

---

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

I started by reading in all the `vehicle` and `non-vehicle` images. there are 8792  cars and 8968  non-cars
 for training and testing, Here is some examples of `vehicle` and `non-vehicle` classes:

[image1]: ./examples/car_not_car.png
![alt text][image1]

Here is an example using the `RGB` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

[image2]: ./examples/HOG_example.png
![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

The 6th cell of CarND-Vehicle-Dection.ipynb show how I settel on the final choice of HOG parameters, I get a smaller data set of car and notcar, small_car number is 1125 and small_notcar is 1125, to investigate the best HOG parameters.

Only to consider the HOG features, experiments are conducted and the result is shown as below:

| NO. | Colorspace | Orient | Pixels perCell | Cells per Block | HOG Channel | Extract Time | Classifier | Accuracy | Train Time |
| :--: | :------: | :----: | :----: | :----: | :------: | :--------: | :--------: | :--------: | :--------: |
|  1 	| RGB 	| 9 	| 8 	| 2 	| ALL 	| 8.44 	|Linear SVC  | 0.9511 	| 5.0503
|  2 	| RGB 	| 12 	| 8 	| 2 	| ALL 	| 8.77 	|Linear SVC  | 0.9489 	| 0.8531
|  3 	| RGB 	| 18 	| 8 	| 2 	| ALL 	| 9.69 	|Linear SVC  | 0.9511 	| 0.9631
|  4 	| RGB 	| 36 	| 8 	| 2 	| ALL 	| 11.49 |Linear SVC  | 0.9578 	| 1.4608
|  5 	| HSV 	| 9 	| 8 	| 2 	| ALL 	| 8.74 	|Linear SVC  | 0.9733 	| 0.3411
|  6 	| HSV 	| 12 	| 8 	| 2 	| ALL 	| 8.98 	|Linear SVC  | 0.9822 	| 0.3944
|  7 	| HSV 	| 18 	| 8 	| 2 	| ALL 	| 9.72 	|Linear SVC  | 0.9778 	| 0.6515
|  8 	| HSV 	| 36 	| 8 	| 2 	| ALL 	| 11.87 |Linear SVC  | 0.9822 	| 1.1518
|  9 	| LUV 	| 9 	| 8 	| 2 	| ALL 	| 9.15 	|Linear SVC  | 0.9578 	| 5.7852
|  10 	| LUV 	| 12 	| 8 	| 2 	| ALL 	| 9.46 	|Linear SVC  | 0.9756 	| 0.4846
|  11 	| LUV 	| 18 	| 8 	| 2 	| ALL 	| 10.01 |Linear SVC  | 0.9711 	| 0.6345
|  12 	| LUV 	| 36 	| 8 	| 2 	| ALL 	| 12.07 |Linear SVC  | 0.9867 	| 1.1297
|  13 	| HLS 	| 9 	| 8 	| 2 	| ALL 	| 8.66 	|Linear SVC  | 0.9778 	| 0.3291
|  14 	| HLS 	| 12 	| 8 	| 2 	| ALL 	| 9.02 	|Linear SVC  | 0.98 	| 0.4273
|  15 	| HLS 	| 18 	| 8 	| 2 	| ALL 	| 9.8 	|Linear SVC  | 0.9756 	| 0.602
|  16 	| HLS 	| 36 	| 8 	| 2 	| ALL 	| 11.91 |Linear SVC  | 0.9778 	| 1.1222
|  17 	| YUV 	| 9 	| 8 	| 2 	| ALL 	| 8.59 	|Linear SVC  | 0.9733 	| 0.3299
|  18 	| YUV 	| 12 	| 8 	| 2 	| ALL 	| 8.93 	|Linear SVC  | 0.9867 	| 0.4075
|  19 	| YUV 	| 18 	| 8 	| 2 	| ALL 	| 9.64 	|Linear SVC  | 0.9778 	| 0.5656
|  20 	| YUV 	| 36 	| 8 	| 2 	| ALL 	| 11.77 |Linear SVC  | 0.9867 	| 1.1095
|  21 	| YCrCb | 9 	| 8 	| 2 	| ALL 	| 8.64 	|Linear SVC  | 0.9756 	| 0.3924
|  22 	| YCrCb | 12 	| 8 	| 2 	| ALL 	| 8.93 	|Linear SVC  | 0.9667 	| 0.3848
|  23 	| YCrCb | 18 	| 8 	| 2 	| ALL 	| 9.66 	|Linear SVC  | 0.9756 	| 0.5941
|  24 	| YCrCb | 36 	| 8 	| 2 	| ALL 	| 11.78 |Linear SVC  | 0.98 	| 1.2059

we can find that RGB color space is not good features for hog to detect vehicles, Time cost increases with the orientation numbers, Orientation num of 12 performorms good at aspect of accuracy and time costs. 

but we still cannot find which  Color space is the best. So I Run the HOG+SVM struction for the whole 8792  cars and 8968  non-cars data set

| NO. | Colorspace | Orient | Pixels perCell | Cells per Block | HOG Channel | Extract Time | Classifier | Accuracy | Train Time |
| :--: | :------: | :----: | :----: | :----: | :------: | :--------: | :--------: | :--------: | :--------: |
|  1 	| RGB 	| 12 	| 8 	| 2 	| ALL 	| 76.16 	|Linear SVC  | 0.9727 	| 30.9131
|  2 	| HSV 	| 12 	| 8 	| 2 	| ALL 	| 78.38 	|Linear SVC  | 0.9828 	| 4.8821
|  3 	| LUV 	| 12 	| 8 	| 2 	| ALL 	| 90.82 	|Linear SVC  | 0.9735 	| 31.413
|  4 	| HLS 	| 12 	| 8 	| 2 	| ALL 	| 77.9 	|Linear SVC  | 0.9831 	| 28.569
|  5 	| YUV 	| 12 	| 8 	| 2 	| ALL 	| 79.44 	|Linear SVC  | 0.9862 	| 5.4692
|  6 	| YCrCb 	| 12 	| 8 	| 2 	| ALL 	| 73.53 	|Linear SVC  | 0.98 	| 19.823

and we find that HOG features perform the best in **YUV Color space** on all vehicles and not vehicels data

so next we want to found the best Parameters and we run the HOG+SVM classifier on the smaller data set to get the intuition.

| NO. | Colorspace | Orient | Pixels perCell | Cells per Block | HOG Channel | Extract Time | Classifier | Accuracy | Train Time |
| :--: | :------: | :----: | :----: | :----: | :------: | :--------: | :--------: | :--------: | :--------: |
|  1 	| YUV 	| 12 	| 8 	| 2 	| ALL 	| 9.18 	|Linear SVC  | 0.98 	| 0.4347
|  2 	| YUV 	| 12 	| 16 	| 2 	| ALL 	| 5.96 	|Linear SVC  | 0.9644 	| 0.8132
|  3 	| YUV 	| 12 	| 32 	| 2 	| ALL 	| 5.36 	|Linear SVC  | 0.9311 	| 0.0665

We can find from tables above that **8 pixels per cell is the best perfmance**



| NO. | Colorspace | Orient | Pixels perCell | Cells per Block | HOG Channel | Extract Time | Classifier | Accuracy | Train Time |
| :--: | :------: | :----: | :----: | :----: | :------: | :--------: | :--------: | :--------: | :--------: |
|  1 	| YUV 	| 12 	| 8 	| 2 	| ALL 	| 8.69 	|Linear SVC  | 0.9778 	| 1.8986
|  2 	| YUV 	| 12 	| 8 	| 4 	| ALL 	| 8.08 	|Linear SVC  | 0.9667 	| 1.0709


We can find from tables above that **2 cells per block will be chosen**

| NO. | spatial | spatial size | hist | hist_bins | Extract Time | Classifier | Accuracy | Train Time |
| :--:| :------:|:---------:| :----------: | :-------: | :----------: | :--------: | :------: | :--------: |
|  1 	| True 	| (32, 32) 	| True 	| 16 	| 9.49 	|Linear SVC  | 0.9822 	| 0.6012
|  2 	| True 	| (32, 32) 	| True 	| 32 	| 9.33 	|Linear SVC  | 0.9933 	| 0.5326
|  3 	| True 	| (32, 32) 	| True 	| 64 	| 9.34 	|Linear SVC  | 0.9844 	| 0.5998
|  4 	| True 	| (32, 32) 	| True 	| 128 	| 9.33 	|Linear SVC  | 0.9933 	| 0.5325
|  5 	| True 	| (64, 64) 	| True 	| 16 	| 9.33 	|Linear SVC  | 0.9844 	| 1.4657
|  6 	| True 	| (64, 64) 	| True 	| 32 	| 9.37 	|Linear SVC  | 0.98 	| 1.3106
|  7 	| True 	| (64, 64) 	| True 	| 64 	| 9.33 	|Linear SVC  | 0.9778 	| 1.723
|  8 	| True 	| (64, 64) 	| True 	| 128 	| 9.33 	|Linear SVC  | 0.98 	| 1.2972
|  9 	| True 	| (128, 128) 	| True 	| 16 	| 9.52 	|Linear SVC  | 0.9844 	| 14.5779
|  10 	| True 	| (128, 128) 	| True 	| 32 	| 10.37 	|Linear SVC  | 0.9667 	| 21.6237
|  11 	| True 	| (128, 128) 	| True 	| 64 	| 14.92 	|Linear SVC  | 0.9733 	| 21.0462
|  12 	| True 	| (128, 128) 	| True 	| 128 	| 10.35 	|Linear SVC  | 0.9756 	| 10.9654

as the table shows below,the best spaitial size and hist_bins is (32, 32) and (32)　repectly.

So the final parameter is that
```python
color_space = 'YUV' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 12  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"
spatial_size = (32, 32) # Spatial binning dimensions
hist_bins = 32    # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = False # Histogram features on or off
hog_feat = True # HOG features on or off
```

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

Since HOG + SVM Frame has achived a hug success in pedestrain detection, so in "Train Classifier" section  of CarND-Vehicle-Dection.ipynb, I trained a linear SVM with the default classifier parameters and using HOG features with color spatial hist features and achives the accuracy of 99.01%

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

In " Sliding Window Search" section, I write a method `find_cars` based on the udacity lesson fuction. `find_cars` is a Hog Sub-sampling Window Search method, it's a combination of HOG feature extraction and a sliding window search to make the process more effcient and time-saving.

the HOG features are extracted for Region of Interest only once, and then features are subsampled according to the size of the window and then fed to the classifier. 

Since the car looks smaller when far away and look larger when it's nearby, we should different scale in the different region. The region is show below:

| y_start_stop | scale |
|:------------------:|:----------------:|
|  (400, 500)  | 1.0, 1.3
|  (410, 500)  | 1.4
|  (420, 556)  | 1.6
|  (430, 556)  | 1.8, 2.0
|  (440, 556)  | 1.9
|  (400, 556)  | 1.3, 2.2
|  (500, 656)  | 3.0


The image below shows the  1st row  of the table's boxes region:

[image3]: ./examples/sliding_window.png
![alt text][image3]

The image below shows the  6st row  of the table's boxes region:

[image31]: ./examples/sliding_window1.png
![alt text][image31]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YUV 3-channel HOG features plus spatially binned color in the feature vector, which provided a nice result. The image below shows the rectangles returned by `find_cars` drawn onto test image in the final implementation.  Here are some example images:

[image4]: ./examples/find_car_result.png
![alt text][image4]

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

[video1]: ./project_video.mp4
Here's a [link to my video result](./result.mp4)

#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

[image51]: ./examples/bboxes_and_heat1.png
[image52]: ./examples/bboxes_and_heat2.png
[image53]: ./examples/bboxes_and_heat3.png
[image54]: ./examples/bboxes_and_heat4.png
[image55]: ./examples/bboxes_and_heat5.png
[image56]: ./examples/bboxes_and_heat6.png

![alt text][image51]
![alt text][image52]
![alt text][image53]
![alt text][image54]
![alt text][image55]
![alt text][image56]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames and the resulting bounding boxes:

[image61]: ./examples/labels_map1.png
[image62]: ./examples/labels_map2.png
[image63]: ./examples/labels_map3.png
[image64]: ./examples/labels_map4.png
[image65]: ./examples/labels_map5.png
[image66]: ./examples/labels_map6.png

![alt text][image61]
![alt text][image62]
![alt text][image63]
![alt text][image64]
![alt text][image65]
![alt text][image66]

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

It takes lots of time to tune the parameter of HOG features, Finnaly I set a small data sets to find the intuition. I think it's is a good choice to do experiements on smaller data sets. 

Particle filter or kalman filter can be added to the future work, Since there are correlations between frame and frame, this will make the work more robust.

SSD or Faster RNN can be a good direction to this project.

