import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
import matplotlib.pyplot as plt

lines = []
images = []
measurements = []
with open('../linux_sim/data/driving_log.csv') as f:
	reader = csv.reader(f)
	for line in reader:
		lines.append(lines)
		images.append(cv2.imread(line[0]))
		measurements.append(float(line[3]))
		images.append(cv2.imread(line[1]))
		measurements.append(float(line[3]) + 0.2)
		images.append(cv2.imread(line[2]))
		measurements.append(float(line[3]) - 0.2)
		# if len(images)>=700:
		# 	break

augmented_images, augmented_measurements = [], []
for image, measurement in zip(images, measurements):
	augmented_images.append(image)
	augmented_measurements.append(measurement)
	augmented_images.append(cv2.flip(image,1))
	augmented_measurements.append(-measurement)

X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)

model = Sequential()
model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape = (160,320,3)))
model.add(Cropping2D(cropping = ((70,25), (0,0))))
model.add(Convolution2D(24,5,5, subsample = (2,2), activation = "relu"))
model.add(Convolution2D(36,5,5, subsample = (2,2), activation = "relu"))
model.add(Convolution2D(48,5,5, subsample = (2,2), activation = "relu"))
# model.add(MaxPooling2D())
model.add(Convolution2D(64,3,3, activation = "relu"))
model.add(Convolution2D(64,3,3, activation = "relu"))
# model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss = 'mse', optimizer = 'adam')
history_object = model.fit(X_train, y_train, validation_split = 0.2, shuffle = True, nb_epoch = 3, verbose = 1)

### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()

model.save('model.h5')
