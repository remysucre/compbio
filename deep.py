
# coding: utf-8

# In[1]:

import numpy as np
import keras
import matplotlib.pyplot as plt


# In[2]:

# read data

X = np.genfromtxt("mnist.data")
Y = np.genfromtxt("mnist.labels")


# In[3]:

# Choose a random image

index = np.random.randint(55000)
image = X[index].reshape(28, 28)
label = Y[index]
print("Label:", np.argmax(label), label)

# Plot

plt.figure(figsize=(2,2))
plt.imshow(image, cmap="gray")
plt.show()


# In[4]:

# split into training & testing sets

split = 50000
trainX, trainY = X[:split], Y[:split]
testX, testY = X[split:], Y[split:]


# In[5]:

# import models & layers

from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, Flatten, Reshape


# In[6]:

# multilayer perceptron

mlp = Sequential()

# input layer 
mlp.add(Dense(15, input_dim=784))
mlp.add(Activation('sigmoid'))

# hidden layer 
mlp.add(Dense(15))
mlp.add(Activation('relu'))

# output layer
mlp.add(Dense(10))
mlp.add(Activation('softmax'))

# Define loss
mlp.compile(optimizer='sgd',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


# In[7]:

# train mlp

mlphistory = mlp.fit(trainX, trainY, epochs=10, verbose=1)


# In[8]:

# convolutional neural net

cnn = Sequential()

# input layer 
cnn.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(28,28,1)))
cnn.add(Activation('sigmoid'))

# hidden layer 
cnn.add(Conv2D(16, kernel_size=(3, 3), strides=(2, 2)))
cnn.add(Activation('relu'))

# output layer
cnn.add(Flatten())
cnn.add(Dense(10))
cnn.add(Activation('softmax'))

# Define loss
cnn.compile(optimizer='sgd',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


# In[9]:

# reshape data into 2D

x_train = trainX.reshape(trainX.shape[0], 28, 28, 1)
x_test = testX.reshape(testX.shape[0], 28, 28, 1)


# In[10]:

# train cnn

cnnhistory = cnn.fit(x_train, trainY, epochs=10, verbose=1)


# In[11]:

# plot mlp loss funciton

f = plt.figure()
plt.plot(mlphistory.history["loss"])
plt.plot(cnnhistory.history["loss"])
plt.show()

f.savefig("loss.pdf", bbox_inches='tight')


# In[12]:

# collect predictions for each network

predictions = mlp.predict(testX)
predictionscnn = cnn.predict(x_test)


# In[13]:

# index = np.random.randint(5000)
# image = testX[index].reshape(28, 28)
# 
# print("MLP Label:", np.argmax(predictions[index]))
# print("CNN Label:", np.argmax(predictionscnn[index]))
# 
# print("MLP result:", predictions[index])
# print("CNN result:", predictionscnn[index])
# 
# plt.figure(figsize=(2,2))
# plt.imshow(image, cmap="gray")
# plt.show()
# 
