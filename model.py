import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import sklearn
from sklearn.utils import shuffle



lines = []
# with open('../driving_log.csv') as csvfile:
with open('../linux_sim/data/driving_log.csv') as f:
	reader = csv.reader(f)
	for line in reader:
		lines.append(line)

train_samples , validation_samples = train_test_split(lines, test_size = 0.2 )

def generator(samples, batch_size = 16):
	num_samples = len(samples)
	while(1):
		shuffle(samples)
		for offset in range(0, num_samples , batch_size):
			batch_samples =samples[offset:offset+batch_size]
			images = []
			measurements = []
			for batch_sample in batch_samples:
				image = batch_sample[0]
				img = cv2.imread(image)
				center_angle = float(batch_sample[3])
				images.append(img)
				measurements.append(center_angle)
				
				left = batch_sample[1]
				right = batch_sample[2]
				
				left_image =cv2.imread(left)
				images.append(left_image)
				measurements.append(center_angle + 0.2)
				
				right_image =cv2.imread(right)
				images.append(right_image)
				measurements.append(center_angle - 0.2)

			augmented_images, augmented_measurements = [], []
			for image , angle in zip(images,measurements):
				augmented_images.append(image)
				augmented_measurements.append(angle)
				augmented_images.append(cv2.flip(image,1))
				augmented_measurements.append(angle*-1.0)
			X_train = np.array(augmented_images)
			y_train = np.array(augmented_measurements)
			yield sklearn.utils.shuffle(X_train , y_train)


train_generator = generator(train_samples ,batch_size = 16)
validation_generator = generator(validation_samples , batch_size = 16)

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
model.add(Dropout(.05))
model.add(Dense(50))
model.add(Dropout(.05))
model.add(Dense(10))
model.add(Dropout(.05))
model.add(Dense(1))

model.compile(loss = 'mse', optimizer = 'adam')
# history_object = model.fit(X_train, y_train, validation_split = 0.2, shuffle = True, nb_epoch = 3, verbose = 1)
history_object=model.fit_generator(train_generator,samples_per_epoch=len(train_samples)*6,validation_data= validation_generator,nb_val_samples=len(validation_samples),nb_epoch=3,verbose=1)


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
