from keras_model import CNN
#import matplotlib.pyplot as plt
import numpy as np 
from sklearn.model_selection import train_test_split
# import GPyOpt as GP
#from bayes_opt import BayesianOptimization, UtilityFunction
#from tqdm import tqdm 
#from itertools import product

from sklearn.gaussian_process import GaussianProcessRegressor as GPR
#import sys
import numpy as np
import matplotlib.pyplot as plt
import GPyOpt
#from torchvision import datasets, transforms, utils
#from sklearn.ensemble import RandomForestClassifier
#from sklearn.model_selection import ParameterSampler, RandomizedSearchCV, cross_val_score
#from scipy.stats import uniform
#import random

from plot2 import plot_perf2, plot_acq_init, plot_acq_full, save_acq_img, load_acq_img, save_pred_img


X = np.load("image_data_gray.npy")
y = np.load("labels.npy")
# X = np.load("image_data_gray.npy")
# y = np.load("labels.npy")

SHAPE = np.shape(X[0])
seed = 42
n_iter = 15
BATCH_SIZE = 125
EPOCHS = 20

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.9, random_state=seed)

domain = [{'name': 'dropRate1', 'type': 'continuous', 'domain': (0.0, 0.5)},
          {'name': 'dropRate2', 'type': 'continuous', 'domain': (0.0, 0.5)}]

def blackBoxFunction(x):
    param = x[0]
    dropRate1 = param[0]
    dropRate2 = param[1]
    model = CNN(dropRate1,dropRate2,input_shape=SHAPE)
    model.train_opt(X_train, y_train, X_test, y_test, batch_size=BATCH_SIZE, epochs = EPOCHS, verbose=0)
    loss, acc = model.evaluate(X_test,y_test)   
    #print(f"Acc: {acc}")
    print(dropRate1, dropRate1)
    #print(-acc)
    return - acc

opt = GPyOpt.methods.BayesianOptimization(f = blackBoxFunction,   # function to optimize
                                              domain = domain,         # box-constrains of the problem
                                              acquisition_type = 'EI' ,      # Select acquisition function MPI, EI, LCB
                                             
                                             )



#opt.acquisition.exploration_weight=0.5
#opt.acquisition.exploration_weight=0.
opt.acquisition.exploration_weight=0.5
NUM = "053"

# Random initialization
opt.run_optimization(max_iter = 0, 
                        verbosity=True,
                        evaluations_file="eval_file_init"+NUM+".txt") 
save_acq_img(opt, "acq_img_init"+NUM+".npy")
save_pred_img(opt, "pred_mean_init"+NUM+"", "pred_var_init"+NUM+"")


# Iter run
opt.run_optimization(max_iter = n_iter, verbosity=True,
                        evaluations_file="eval_file_full"+NUM+".txt", eps = 0.001) 
save_acq_img(opt, "acq_img_full"+NUM+".npy")
save_pred_img(opt, "pred_mean_full"+NUM+"", "pred_var_full"+NUM+"")

# Print output
x_best = opt.X[np.argmin(opt.Y)]
print("The best parameters obtained: dropRate1=" + str(x_best[0]) + ", dropRate2=" + str(x_best[1]))
