# pballdetection
Detection and classification of 3D balls in a 2D image

Divided into 2 subtasks, one for detecting the locations of the balls in the image, one for classifying the ball. Used for implementation in dispenser systems.

Design:
To detect the balls in the image, or any object in an real-world image, if the objects are not occluded, the output will be in coordinates of the bounding boxes.
For every image, the dimensions are constant, and the output bounding box coordinates are real number pairs in [0, 1] x [0, 1], to allow scaling to different image dimensions.
Initial idea is to use an RNN with a CNN on the input to output the bounding boxes in a vector. The loss can be IOU or another metric.

Given the model to detect correct bounding boxes, the cropped regions with the boxes are used to feed into a CNN to classify them into a class. Standard cross-entropy 
can be used to train the classification portion.

Alternative is to output the location and classify the objects in an end-to-end model, and then compare the aggregate results with the correct answer. This approach
is difficult in establishing a metric to quantify a the correctness of a classification. 

Data synthesis:
Training data is not given, and for a supervised task we generate our own data in the hope that such a model can transfer to classification in a real image.
We know the background as a base setting and can obtain images of the objects from various angles to build a representation of the 3D object.
To generate a sample, we randomly sample from a uniform joint distribution to determine which balls will be in the image.
Generate an image for each ball that will appear based on the given images and perturbations. (May have to use a GAN).
Sample random coordinates on the background to determine where each ball is situated, and superimpose their images on the background.
The corresponding tag for each image is the ball and their bounding boxes.

Train the bounding box classifier and the object classifier separately and use together in an end-to-end model.

