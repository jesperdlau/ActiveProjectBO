
import matplotlib.pyplot as plt
import numpy as np 
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.models import Sequential
from keras import layers

class CNN():
    def __init__(self,n_extra_layers=0,epochs=5,loss_func="sparse_categorical_crossentropy",act="relu",opt="adam"):
        self.epochs = epochs
        self.model = Sequential()
        self.model.add(layers.Conv2D(80,(3,3),activation=act,input_shape=(80,80,3)))
        self.model.add(layers.MaxPooling2D((2,2), padding='same'))

        for i in range(n_extra_layers):
            self.model.add(layers.Conv2D(160,(3,3),activation=act))
            self.model.add(layers.MaxPooling2D((2,2), padding='same'))

        # self.model.add(layers.MaxPooling2D((2,2), padding='same'))
        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(160,activation=act))
        self.model.add(layers.Dense(2,activation="softmax"))
        self.model.compile(optimizer=opt,loss=loss_func,metrics=["accuracy"])

    def train(self,X_train,y_train,X_test,y_test):
        self.model.fit(X_train,y_train,self.epochs,validation_data=(X_test,y_test))

    def evaluate(self,X_test,y_test):
        loss,acc = self.model.evaluate(X_test,y_test)
        return acc

X = np.load("image_data.npy")
y = np.load("labels.npy")

# print(X[0].shape)
# train_test_split with shuffle
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# plt.imshow(X_train[0])
# plt.show()

model = CNN(1,5)
model.train(X_train,y_train,X_test,y_test)

acc_train = model.evaluate(X_train,y_train)
acc_test = model.evaluate(X_test,y_test)

print("Train accuracy = ", acc_train)
print("Test accuracy = ",acc_test)

