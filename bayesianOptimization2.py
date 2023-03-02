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

from plot2 import acqfunc, plot_perf, plot_arrow


X = np.load("image_data_gray.npy")[:200]
y = np.load("labels.npy")[:200]
# X = np.load("image_data_gray.npy")
# y = np.load("labels.npy")

SHAPE = np.shape(X[0])
seed = 42
n_iter = 10
BATCH_SIZE = 100
EPOCHS = 20

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=seed)

# define the dictionary for GPyOpt
# domain = [{'name': 'dropRate1', 'type': 'discrete', 'domain': tuple(np.arange(0, 0.5, 0.01))},
#           {'name': 'dropRate2', 'type': 'discrete', 'domain': tuple(np.arange(0, 0.5, 0.01))}]

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
    return acc


opt = GPyOpt.methods.BayesianOptimization(f = blackBoxFunction,   # function to optimize
                                              domain = domain,         # box-constrains of the problem
                                              acquisition_type = 'EI' ,      # Select acquisition function MPI, EI, LCB
                                              


                                             )

opt.acquisition.exploration_weight=0.5

for n in range(n_iter):
    opt.run_optimization(max_iter = 1, 
                        verbosity=True,
                        #report_file="report_file_"+str(n)+".txt",
                        evaluations_file="eval_file_"+str(n)+".txt", 
                        #models_file="models_file_"+str(n)+".txt"
                        ) 
    acqfunc(opt, "acq_plot_"+str(n)+".png")

plot_perf(opt, "performance_plot.png")
plot_arrow("eval_file_"+str(n_iter-1)+".txt", "arrow_plot.png")

x_best = opt.X[np.argmin(opt.Y)]
print("The best parameters obtained: dropRate1=" + str(x_best[0]) + ", dropRate2=" + str(x_best[1]))




# print(opt.model.input_dim)
# print(opt.model.get_model_parameters_names())
# print(opt.model.get_model_parameters())

#acqfunc(opt, "acq.png")


