# **Behavioral Cloning** 

## Writeup


---



[//]: # (Image References)

[image1]: ./loss.png "Loss"
[image2]: ./center.png "Center"
[image3]: ./recovery1.jpg "Recovery1 Image"
[image4]: ./recovery2.jpg "Recovery2 Image"
[image5]: ./recovery3.jpg "Recovery3 Image"
[image6]: ./original.jpg "Normal Image"
[image7]: ./flipped.png "Flipped Image"



My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy
My model (model.py 64-81) is based on the architecture provided in the Nvidia paper.Initially I use a keras lambda layer to normalize the image values and a Cropping2D layer for cropping the image to filter out the region of interest. It then has 5 convolutional layers with relu activation followed by three fully connected layers. The first three convolution layers use 5x5 kernels and the other 2 layers use 3x3 kernel. It has depths between 24 and 64 (model.py lines 67-72)


#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 76, 78, 80). 

The model was trained and validated on different data sets(model.py line 17) to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 83).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road, and also driving in the reverse direction as there were more left turns than right turns. Even then the model had some issues completing a lap. I solved this problem by retraining it, especially on some sharp turns.

The images for the training data were from the left, right and center camera. So, in addition to the center camera image I also used the left and the right camera images and gave them a correction factor of 0.2 to the corresponding steering angle.I also created more data by flipping the images and taking the negative of the corresponding steering angle.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture, given the input images, was to get the output as the steering angle of the car that would help it navigate the track.

My first step was to use a convolution neural network model similar to the Nvidia paper architecture discussed in the lectures. I thought this model might be appropriate because it worked reasonably well when the instructor showed a demo of it in the lectures and it had sufficient convolutional layers to extract the necessary features.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model by adding dropout layers. This solved the problem to some extent.

Then I tried reducing the number of epochs it trained on from 6 to 3 and this worked for me. The training and validation loss were comparable.

![alt text][image1]

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track. To improve the driving behavior in these cases, I retrained the model by just driving along those spots.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 64-81) consisted of a convolution neural network with the following layers and layer sizes:

| Layer         		|     Description	        					                | 
|:---------------------:|:---------------------------------------------:                | 
|Input                  | RGB image (160,320,3)                                         |
|Lambda |                          Normalize the image pixels, outputs (None, 160, 320, 3)|                                
|Cropping2D |                                  Crops the image, outputs (None, 65, 320, 3)|                                
|Convolution2D | 5X5 kernel, (2,2) sampling, relu activation, outputs(None, 31, 158, 24)    |
|Convolution2D | 5X5 kernel, (2,2) sampling, relu activation, outputs(None, 14, 77, 36)    |
|Convolution2D | 5X5 kernel, (2,2) sampling, relu activation, outputs(None, 5, 37, 48)    |
|Convolution2D | 3X3 kernel, relu activation, outputs(None, 3, 35, 64)    |
|Convolution2D | 3X3 kernel, relu activation, outputs(None, 1, 33, 64)    |
|Flatten                                                         |outputs (None, 2112)              |
|Dense                                                           |outputs (None, 100)                |
|Dropout (0.05)                                                  |outputs (None, 100)                |              
|Dense                                                           |outputs (None, 50)          |
|Dropout (0.05)                                                 |outputs (None, 50)                |
|Dense                                                          | outputs (None, 10)          |
|Dropout (0.05)                                                 |outputs (None, 10)                |
|Dense                                                          | (None, 1)            |    




#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to position itself on the center of the track. These images show what a recovery from the left side of the track looks like:

![alt text][image3]

![alt text][image4]

![alt text][image5]


To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

I randomly shuffled the data set and put 20% of the data into a validation set. 

I then preprocessed this data by normalising the pixel values. Then I cropped the top half of the image as it just had trees and the sky which are not necessary to train the model.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 3, which was arrived at by trial and error. I used an adam optimizer so that manually training the learning rate wasn't necessary.
