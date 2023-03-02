from keras_model import CNN
import matplotlib.pyplot as plt
import numpy as np 
from tqdm import tqdm 
import random 
from sklearn.model_selection import train_test_split

# load data
X = np.load("image_data_gray.npy")
y = np.load("labels.npy")
X = np.load("image_data_gray.npy")[:200]
y = np.load("labels.npy")[:200]

seed = 42
n_iter = 10
SHAPE = np.shape(X[0])
BATCH_SIZE = 100
EPOCHS = 20

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=seed)


results = []

for i in tqdm(range(n_iter)):
    dropRate1 = random.uniform(0.0,0.5)
    dropRate2 = random.uniform(0.0,0.5)
    print("Droprate1: ", dropRate1)
    print("Droprate2: ", dropRate2)
    model = CNN(dropRate1,dropRate2, SHAPE)
    model.train_opt(X_train,y_train,X_test,y_test,
                    batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=2)
    loss, acc = model.evaluate(X_test,y_test)
    results.append(np.array([i,acc,dropRate1,dropRate2]))

np.save("randomSearchResults.npy", np.array(results))

for iteration in results:
    print(iteration)
    




