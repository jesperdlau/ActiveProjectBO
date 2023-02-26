from keras_model import CNN

import matplotlib.pyplot as plt
import numpy as np 
from sklearn.model_selection import train_test_split
from bayes_opt import UtilityFunction
from matplotlib import gridspec
# %matplotlib inline
from bayes_opt import BayesianOptimization
from tqdm import tqdm 
# Load data
# X = np.load("image_data.npy")
X = np.load("image_data_gray.npy")
y = np.load("labels.npy")
SHAPE = np.shape(X[0])
seed = 42
n_iter = 15
BATCH_SIZE = 100
EPOCHS = 20

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=seed)

def blackBoxFunction(dropRate1,dropRate2):
    model = CNN(dropRate1,dropRate2,input_shape=SHAPE)
    model.train_opt(X_train, y_train, X_test, y_test, batch_size=BATCH_SIZE, epochs = EPOCHS, verbose=0)
    loss, acc = model.evaluate(X_test,y_test)   
    return acc

# # with bayes_opt simple approach
bounds2D = {"dropRate1":(0.0,0.5),"dropRate2":(0.0,0.5)}
optimizer = BayesianOptimization(f=blackBoxFunction, pbounds=bounds2D, verbose=2, random_state=seed)
print("###########Starting optimization##############")
optimizer.maximize(init_points = 5,n_iter = n_iter)

print(optimizer.max)


# ### fra: https://github.com/fmfn/BayesianOptimization/blob/master/examples/visualization.ipynb
# def posterior(optimizer, x_obs, y_obs, grid):
#     optimizer._gp.fit(x_obs, y_obs)

#     mu, sigma = optimizer._gp.predict(grid, return_std=True)
#     return mu, sigma

# ### fra: https://github.com/fmfn/BayesianOptimization/blob/master/examples/visualization.ipynb
# def plot_gp(optimizer, x, y):
#     fig = plt.figure(figsize=(16, 10))
#     steps = len(optimizer.space)
#     fig.suptitle(
#         'Gaussian Process and Utility Function After {} Steps'.format(steps),
#         fontdict={'size':30}
#     )
    
#     gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1]) 
#     axis = plt.subplot(gs[0])
#     acq = plt.subplot(gs[1])
    
#     x_obs = np.array([[res["params"]["x"]] for res in optimizer.res])
#     y_obs = np.array([res["target"] for res in optimizer.res])
    
#     mu, sigma = posterior(optimizer, x_obs, y_obs, x)
#     axis.plot(x, y, linewidth=3, label='Target')
#     axis.plot(x_obs.flatten(), y_obs, 'D', markersize=8, label=u'Observations', color='r')
#     axis.plot(x, mu, '--', color='k', label='Prediction')

#     axis.fill(np.concatenate([x, x[::-1]]), 
#               np.concatenate([mu - 1.9600 * sigma, (mu + 1.9600 * sigma)[::-1]]),
#         alpha=.6, fc='c', ec='None', label='95% confidence interval')
    
#     axis.set_xlim((-2, 10))
#     axis.set_ylim((None, None))
#     axis.set_ylabel('f(x)', fontdict={'size':20})
#     axis.set_xlabel('x', fontdict={'size':20})
    
#     utility_function = UtilityFunction(kind="ucb", kappa=5, xi=0)
#     utility = utility_function.utility(x, optimizer._gp, 0)
#     acq.plot(x, utility, label='Utility Function', color='purple')
#     acq.plot(x[np.argmax(utility)], np.max(utility), '*', markersize=15, 
#              label=u'Next Best Guess', markerfacecolor='gold', markeredgecolor='k', markeredgewidth=1)
#     acq.set_xlim((-2, 10))
#     acq.set_ylim((0, np.max(utility) + 0.5))
#     acq.set_ylabel('Utility', fontdict={'size':20})
#     acq.set_xlabel('x', fontdict={'size':20})
    
#     axis.legend(loc=2, bbox_to_anchor=(1.01, 1), borderaxespad=0.)
#     acq.legend(loc=2, bbox_to_anchor=(1.01, 1), borderaxespad=0.)


# plot_gp(optimizer, )