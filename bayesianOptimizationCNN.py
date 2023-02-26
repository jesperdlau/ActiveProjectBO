from keras_model import CNN
import matplotlib.pyplot as plt
import numpy as np 
from sklearn.model_selection import train_test_split
# import GPyOpt as GP
from bayes_opt import BayesianOptimization, UtilityFunction
from tqdm import tqdm 
from itertools import product

# Load data
# X = np.load("image_data.npy")
X = np.load("image_data_gray.npy")
y = np.load("labels.npy")
SHAPE = np.shape(X[0])
seed = 42
n_iter = 15
BATCH_SIZE = 80
EPOCHS = 20

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=seed)

def blackBoxFunction(dropRate1,dropRate2):
    model = CNN(dropRate1,dropRate2,input_shape=SHAPE)
    model.train_opt(X_train, y_train, X_test, y_test, batch_size=BATCH_SIZE, epochs = EPOCHS, verbose=0)
    loss, acc = model.evaluate(X_test,y_test)   
    return acc

# with bayes_opt simple approach
bounds2D = {"dropRate1":(0.0,0.5),"dropRate2":(0.0,0.5)}
optimizer = BayesianOptimization(f=blackBoxFunction, pbounds=bounds2D, verbose=2, random_state=seed)
print("###########Starting optimization##############")
#optimizer.maximize(init_points = 5, n_iter = n_iter)


### Broken down optimization
utility_list = []
utility_function = UtilityFunction(kind="ucb", kappa=5, xi=0)

# Random initialization
optimizer.maximize(init_points = 5, n_iter = 0)
print("Random done")

# Iter loop
x1 = np.linspace(0, 0.5, num=101)
x2 = np.linspace(0, 0.5, num=101)
x1x2 = np.array(list(product(x1, x2)))
for i in range(n_iter):
    optimizer.maximize(init_points = 0, n_iter = 1)
    utility = utility_function.utility(x1x2, optimizer._gp, 0)
    X0p, X1p = x1x2[:,0].reshape(101,101), x1x2[:,1].reshape(101,101)
    utility = np.reshape(utility, (101,101))
    utility_list.append(utility)

# Save utility
np.save("utility.npy", np.array(utility_list))

# Plot
fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111)
ax.pcolormesh(X0p, X1p, utility)
#ax.contourf(X0p, X1p, utility)
plt.colorbar()
plt.show()


print()


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
