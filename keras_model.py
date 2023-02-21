
import matplotlib.pyplot as plt
import numpy as np 
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.models import Sequential
from keras import layers
from keras import callbacks

class CNN():
    def __init__(self,dropRate1=0.1,dropRate2=0.1,n_extra_layers=0,
    loss_func="sparse_categorical_crossentropy",
    act="relu",opt="adam"):
        self.model = Sequential()
        self.model.add(layers.Conv2D(80,(3,3),activation=act,input_shape=(80,80,3)))
        self.model.add(layers.MaxPooling2D((2,2), padding='same'))

        for i in range(n_extra_layers):
            self.model.add(layers.Conv2D(160,(3,3),activation=act))
            self.model.add(layers.MaxPooling2D((2,2), padding='same'))

        # self.model.add(layers.MaxPooling2D((2,2), padding='same'))
        self.model.add(layers.Dropout(dropRate1))
        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(160,activation=act))
        self.model.add(layers.Dropout(dropRate2))
        self.model.add(layers.Dense(2,activation="softmax"))
        self.model.compile(optimizer=opt,loss=loss_func,metrics=["accuracy"])

    def train(self,X_train,y_train,X_test,y_test,epochs):
        self.model.fit(X_train,y_train,epochs,validation_data=(X_test,y_test))
    
    def train_opt(self,X_train,y_train,X_test,y_test):
        earlystopping = callbacks.EarlyStopping(monitor ="val_loss", 
                                        mode ="min", patience = 5, 
                                        restore_best_weights = True)
        self.model.fit(X_train, y_train, batch_size = 128, 
                    epochs = 25, validation_data =(X_test, y_test), 
                    callbacks =[earlystopping])

    def evaluate(self,X_test,y_test):
        loss,acc = self.model.evaluate(X_test,y_test)
        return loss, acc

X = np.load("image_data.npy")
y = np.load("labels.npy")

# print(X[0].shape)
# train_test_split with shuffle
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# plt.imshow(X_train[0])
# plt.show()

model = CNN()
# model.train(X_train,y_train,X_test,y_test,10)
model.train_opt(X_train,y_train,X_test,y_test)
model.model.summary()

loss_train, acc_train = model.evaluate(X_train,y_train)
loss_test, acc_test = model.evaluate(X_test,y_test)

print("Train accuracy = ", acc_train)
print("Test accuracy = ",acc_test)

