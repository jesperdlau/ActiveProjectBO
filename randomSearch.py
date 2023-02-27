from keras_model import CNN
import matplotlib.pyplot as plt
import numpy as np 
from tqdm import tqdm 
import random 
from sklearn.model_selection import train_test_split

n_iter = 10
seed = 42

# load data
X = np.load("image_data.npy")
y = np.load("labels.npy")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=seed)

results = []

for i in tqdm(range(n_iter)):
    dropRate1 = random.uniform(0.0,0.5)
    dropRate2 = random.uniform(0.0,0.5)
    print("Droprate1: ")
    print("Droprate2: ")
    model = CNN(dropRate1,dropRate2)
    model.train_opt(X_train,y_train,X_test,y_test,verbose=2)
    loss, acc = model.evaluate(X_test,y_test)
    results.append((i,acc,dropRate1,dropRate2))

for iteration in results:
    print(iteration)



