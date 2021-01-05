import skimage.transform
import numpy as np
from sklearn.model_selection import train_test_split
import keras
from keras.utils import np_utils
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras import initializers
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.models import Sequential


filename = 'hiragana.npz'
char = 71
row, col = 48,48
writ = 160

hira = np.load(filename)['arr_0'].reshape([-1,127,128]).astype(np.float32) / 15

X = np.zeros([char*writ,row,col], dtype=np.float32)

for i in range(char*writ):
    X[i] = skimage.transform.resize(hira[i], (row,col))

arr = np.arange(char)
y = np.repeat(arr,writ)

train_X, test_X, train_y, test_y = train_test_split(X,y, test_size=0.2)

if K.image_data_format() == 'channels_first':
    train_X = train_X.reshape(train_X.shape[0], 1, row, col)
    test_X = test_X.reshape(test_X.shape[0], 1, row, col)
    shape = (1, row, col)
else:
    train_X = train_X.reshape(train_X.shape[0], row, col, 1)
    test_X = test_X.reshape(test_X.shape[0], row, col, 1)
    shape = (row, col, 1)

train_y = np_utils.to_categorical(train_y, writ)
test_y = np_utils.to_categorical(test_y, writ)

datagen = ImageDataGenerator(rotation_range=15, zoom_range=0.20)
datagen.fit(train_X)


model = keras.Sequential([
  keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=shape),
  keras.layers.MaxPooling2D(2,2),
  keras.layers.Conv2D(64, (3,3), activation='relu'),
  keras.layers.MaxPooling2D(2,2),
  keras.layers.Conv2D(64, (3,3), activation='relu'),
  keras.layers.MaxPooling2D(2,2),
  keras.layers.Flatten(),
  keras.layers.Dropout(0.5),
  keras.layers.Dense(1024, activation='relu'),
  keras.layers.Dense(writ, activation="softmax")
])

model.summary()

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit_generator(datagen.flow(train_X, train_y, batch_size=16), 
                    steps_per_epoch=train_X.shape[0]//16,
                    epochs=30, 
                    validation_data=(test_X, test_y))


#save model
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("hiragana_w.h5")
model.save("hiragana.h5")
print("Models saved.")