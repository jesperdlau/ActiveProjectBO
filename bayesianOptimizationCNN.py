from keras_model import CNN

import matplotlib.pyplot as plt
import numpy as np 
from sklearn.model_selection import train_test_split
# import GPyOpt as GP
from bayes_opt import BayesianOptimization
from tqdm import tqdm 
# Load data
X = np.load("image_data.npy")
y = np.load("labels.npy")
seed = 42
n_iter = 5

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=seed)

def blackBoxFunction(dropRate1,dropRate2):
    model = CNN(dropRate1,dropRate2)
    model.train_opt(X_train,y_train,X_test,y_test,verbose=0)
    loss, acc = model.evaluate(X_test,y_test)   
    return acc

# with bayes_opt simple approach
bounds2D = {"dropRate1":(0.0,0.5),"dropRate2":(0.0,0.5)}
optimizer = BayesianOptimization(f=blackBoxFunction,pbounds=bounds2D,verbose=2,random_state=seed)
print("###########Starting optimization##############")
optimizer.maximize(init_points = 5,n_iter = n_iter)


# with bayes_opt more manual approach
# bounds2D = {"dropRate1":[0.0,0.5],"dropRate2":[0.0,0.5]}
# optimizer = BayesianOptimization(f=None,pbounds=bounds2D,verbose=2,random_state=seed)

# for i in tqdm(n_iter):
#     next_point = optimizer.suggest(utility)


#################### virker ikke ################
# with GPyOpt
# bounds2D = [{"dropRate1":"x","type":"continuous","domain":(0.0,0.5)},
#             {"dropRate2":"y","type":"continuous","domian":(0.0,0.5)}]

# n_iter = 20

# bo = GP.methods.BayesianOptimization(blackBoxFunction,domain=bounds2D,acquisition_type="EI")
# bo.run_optimization(max_iter = n_iter)
