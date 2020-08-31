import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout,Dense,Convolution2D,MaxPooling2D,Flatten
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

(X_train,y_train),(X_test,y_test) = mnist.load_data()

for i in range(6):
  plt.subplot(int('23'+str(i+1)))
  plt.imshow(X_train[i],cmap = 'gray')

print(y_train[:6])

X_train = X_train.reshape(X_train.shape[0],28,28,1).astype('float32')
X_test = X_test.reshape(X_test.shape[0],28,28,1).astype('float32')

y_train_cat = to_categorical(y_train)
y_test_cat = to_categorical(y_test)

X_train = X_train/255
X_test = X_test/255

model = Sequential()
model.add(Convolution2D(32,(3,3),activation='relu',input_shape = (28,28,1)))
model.add(Convolution2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(256,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10,activation='softmax'))

model.compile(loss = 'categorical_crossentropy',optimizer = 'adam',metrics = ['accuracy'])

model.summary()

model.fit(X_train,y_train_cat,validation_data=(X_test,y_test_cat),epochs=10,batch_size=200,verbose=2)

model.save('mnist_digit.h5')

