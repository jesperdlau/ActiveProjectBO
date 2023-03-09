import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from skimage import io

def load_eval(eval_file):
    # Import data
    df = pd.read_csv(eval_file, delimiter = "\t", index_col="Iteration")
    df.columns = ["acc", "dropRate1", "dropRate2"]
    df.index.name = "Iteration"
    df.index = df.index.astype('int8')
    #print(df)
    print("Loaded eval file")
    df["acc"] = -1*df["acc"]
    return df.round(4)

def load_random(random_file):
    arr = np.load(random_file)
    df = pd.DataFrame(arr, columns=["Iteration", "acc", "dropRate1", "dropRate2"])
    df.set_index("Iteration", inplace=True)
    df.index = df.index.astype('int8')
    df.index = df.index + 1
    print("loaded random")
    return df.round(4)

def save_acq_img(optimizer, fileout):
    xax = np.arange(0, 0.5, 0.001)
    yax = np.arange(0, 0.5, 0.001)
    pgrid = np.array(np.meshgrid(xax, yax,[1],[0],[1],[0],indexing='ij'))
    print(pgrid.reshape(6,-1).T.shape)
    #we then unfold the 4D array and simply pass it to the acqusition function
    acq_img = optimizer.acquisition.acquisition_function(pgrid.reshape(6,-1).T)
    #it is typical to scale this between 0 and 1:
    acq_img = (-acq_img - np.min(-acq_img))/(np.max(-acq_img - np.min(-acq_img)))
    #then fold it back into an image and plot
    acq_img = acq_img.reshape(pgrid[0].shape[:2])
    np.save(fileout, acq_img)
    print("acquisition img saved to numpy file")
    #return acq_img

def save_pred_img(optimizer, fileoutmean, fileoutvar):
    xax = np.arange(0, 0.5, 0.001)
    yax = np.arange(0, 0.5, 0.001)
    pgrid = np.array(np.meshgrid(xax, yax,[1],[0],[1],[0],indexing='ij'))
    print(pgrid.reshape(6,-1).T.shape)
    #we then unfold the 4D array and simply pass it to the acqusition function
    #acq_img = optimizer.acquisition.acquisition_function(pgrid.reshape(6,-1).T)
    pred_mean, pred_var = optimizer.model.model.predict(pgrid.reshape(6,-1).T)
    #it is typical to scale this between 0 and 1:
    np.save(fileoutmean, pred_mean)
    np.save(fileoutvar, pred_var)
    print("prediction images saved to numpy files")

def format_prediction(fileoutmean, fileoutvar):
    xax = np.arange(0, 0.5, 0.001)
    yax = np.arange(0, 0.5, 0.001)
    pgrid = np.array(np.meshgrid(xax, yax,[1],[0],[1],[0],indexing='ij'))
    pred_mean = load_acq_img(fileoutmean)
    pred_var = load_acq_img(fileoutvar)

    pred_mean = (-pred_mean - np.min(-pred_mean))/(np.max(-pred_mean - np.min(-pred_mean)))
    #pred_var = (-pred_var - np.min(-pred_var))/(np.max(-pred_var - np.min(-pred_var))) 
    #then fold it back into an image and plot
    pred_mean = pred_mean.reshape(pgrid[0].shape[:2])
    pred_var = pred_var.reshape(pgrid[0].shape[:2])

    np.save(fileoutmean, pred_mean)
    np.save(fileoutvar, pred_var)
    print("formatted prediction images saved to numpy files")

def load_acq_img(filein):
    acq_img = np.load(filein)
    return acq_img

def scale_range (input, min, max):
    input += -(np.min(input))
    input /= np.max(input) / (max - min)
    input += min
    return input

def plot_random(random_file, file_out):
    df = load_random(random_file)

    #Plot accuracy points
    sns.scatterplot(data=df, x="dropRate1", y="dropRate2", size="acc", 
                    sizes=(50, 200), color="black", marker="o", legend=False)
    plt.savefig(file_out)
    plt.close()
    #plt.show()

### Plot acquisition function 2d ONLY Random point - with overlay
def plot_acq_init(acq_file, file, evalfile, title):
    xax = np.arange(0, 0.5, 0.001)
    yax = np.arange(0, 0.5, 0.001)
    acq_img = load_acq_img(acq_file)
    # Acc data
    df = load_eval(evalfile)
    df_len = len(df)

    plt.figure()
    plt.imshow(acq_img.T, origin='lower',extent=[xax[0],xax[-1],yax[0],yax[-1]])
    plt.colorbar()

    # Plot accuracy points
    # sns.scatterplot(data=df, x="dropRate1", y="dropRate2", size="acc", 
    #                 sizes=(40, 80), color="red", marker="*", legend=False)
    # Text labels
    for i in range(1, df_len+1):
        plt.text(x=df['dropRate1'][i]+0.002, y=df['dropRate2'][i]+0.002, s=f"{df['acc'][i]:.3f}", 
                fontdict=dict(color="black", size=8), bbox=dict(boxstyle="round", fc="w"))
        
    plt.xlabel('dropRate1')
    plt.ylabel('dropRate2')
    plt.title(title)
    plt.savefig(file)
    print("saved img init")
    plt.close()

### Plot acquisition function 2d - with overlay
def plot_acq_full(acq_file, file, evalfile, title, scale=False):
    xax = np.arange(0, 0.5, 0.001)
    yax = np.arange(0, 0.5, 0.001)
    acq_img = load_acq_img(acq_file)
    df = load_eval(evalfile)
    if scale == True: acq_img = scale_range(acq_img, min(df["acc"]), max(df["acc"])) 

    plt.figure(figsize=(5,4))
    plt.imshow(acq_img.T, origin='lower',extent=[xax[0],xax[-1],yax[0],yax[-1]])
    plt.colorbar()

    # Plot INIT random points
    sns.scatterplot(data=df.iloc[:5], x="dropRate1", y="dropRate2", size="acc", 
                    sizes=(100, 200), color="red", marker="*", legend=False)

    # Plot ITER sequential points
    for i in range(6, len(df)+1):
        plt.text(x=df['dropRate1'][i], y=df['dropRate2'][i], s=str(i-5), 
                fontdict=dict(color="black", size=10), bbox=dict(boxstyle="round", fc="w"))
        
    # Arrows
    for i in range(6,len(df)):
        x, y = df['dropRate1'][i], df['dropRate2'][i]
        dx, dy = (df['dropRate1'][i+1] - x), (df['dropRate2'][i+1] - y)
        plt.arrow(x, y, dx, dy, 
                width = 0.0005, head_width = 0, linestyle='dotted')
        
    plt.xlabel('dropRate1')
    plt.ylabel('dropRate2')
    plt.title(title)
    plt.savefig(file)
    print("saved img full")
    plt.close()

def plot_pred_random_full(acq_file, file_out, eval_file, randomfile, title, scale=True):
    xax = np.arange(0, 0.5, 0.001)
    yax = np.arange(0, 0.5, 0.001)
    acq_img = load_acq_img(acq_file)
    df_random = load_random(randomfile)
    df = load_eval(eval_file)
    if scale == True: acq_img = scale_range(acq_img, min(df["acc"]), max(df["acc"])) 

    plt.figure(figsize=(5,4))
    plt.imshow(acq_img.T, origin='lower',extent=[xax[0],xax[-1],yax[0],yax[-1]])
    plt.colorbar()

    sns.scatterplot(data=df_random, x="dropRate1", y="dropRate2", size="acc", 
                    sizes=(50, 200), color="black", marker="o", legend=False)
    
    for i in range(1, len(df)+1):
        plt.text(x=df_random['dropRate1'][i]-0.04, y=df_random['dropRate2'][i]-0.03, s=f"{df_random['acc'][i]:.2f}", 
                fontdict=dict(color="black", size=8), bbox=dict(boxstyle="round", fc="w"))
    
    plt.xlabel('dropRate1')
    plt.ylabel('dropRate2')
    plt.title(title)
    plt.savefig(file_out)
    print("saved img")
    plt.close()

### Plot performance over time
def plot_perf2(filein, random, fileto):
    df = load_eval(filein)
    rand = load_random(random)
    y = df['acc']
    y_cum = np.maximum.accumulate(df['acc']).ravel()
    y_rand = rand['acc']
    y_rand_cum = np.maximum.accumulate(y_rand).ravel()
    max_len = max(len(y_cum), len(y_rand))
    xs = np.arange(1,max_len+1,1)
    plt.plot(xs, y_rand_cum, 'o-', color = 'red', label='Best Random Search')
    plt.plot(xs, y_cum, 'o-', color = 'blue', label='Best Bayesian Optimization')
    plt.plot(xs, y_rand, 'x-', color = 'red', label='Random Search', alpha=0.3)
    plt.plot(xs, y, 'x-', color = 'blue', label='Bayesian Optimization', alpha=0.3)
    plt.legend()
    plt.xticks(np.arange(21))
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.title('Comparison between Random Search and Bayesian Optimization')
    plt.savefig(fileto)
    print("saved performance-over-time plot")
    plt.close()



if __name__ == "__main__":
    #plot_arrow("eval_file_4.txt", "arrow_plot.png")
    #plot_arrow5("eval_file_4.txt", "arrow_plot5.png")


 
    # Using all data at 0.9 test proportion. 125 batch size, 20 max epochs. Seed 42. 
    # 052: weight 0.05 - 15 iter. Good init, good traversal, but uniform full OR was it 0.5? Run again..
    # 2: weight 2 - 15 iter but stops early after 1 iter.. Run again with lower eps?
    # 11: weight 1 - 15 iter - Good iter, uniform full
    # 01: weight 0.1 - Quickly converges. Not interesting
    # 0.5, 0.1?
    # 0.53: weight 0.5 - Very certain init but traversal is very explorative! Beautiful full acq. 

    NUM = "053"

    # Format

    #format_prediction("pred_mean_full"+NUM+".npy", "pred_var_full"+NUM+".npy")
    #format_prediction("pred_mean_init"+NUM+".npy", "pred_var_init"+NUM+".npy")

    #plot_perf2("eval_file_full"+NUM+".txt", "randomSearchResults250.npy",  "performance_plot"+NUM+".png")

    # plot_acq_init("acq_img_init"+NUM+".npy", "plot_acq_init"+NUM+".png", "eval_file_init"+NUM+".txt", title="Acquisition Function")
    # plot_acq_full("acq_img_full"+NUM+".npy", "plot_acq_full"+NUM+".png", "eval_file_full"+NUM+".txt", title="Acquisition Function")

    # plot_acq_full("pred_mean_full"+NUM+".npy", "pred_mean_full"+NUM+".png", "eval_file_full"+NUM+".txt", title="Mean Prediction Function", scale=True)
    # plot_acq_init("pred_mean_init"+NUM+".npy", "pred_mean_init"+NUM+".png", "eval_file_init"+NUM+".txt", title="Mean Prediction Function")

    # plot_acq_full("pred_var_full"+NUM+".npy", "pred_var_full"+NUM+".png", "eval_file_full"+NUM+".txt", title="Var Prediction Function")
    # plot_acq_init("pred_var_init"+NUM+".npy", "pred_var_init"+NUM+".png", "eval_file_init"+NUM+".txt", title="Var Prediction Function")


    # Mean pred with random overlay
    plot_pred_random_full("pred_mean_full"+NUM+".npy", "Plot_pred_random"+NUM+".png", "eval_file_full"+NUM+".txt",  "randomSearchResults250.npy", title="Mean Prediction Function, Random Overlay", scale=True)

