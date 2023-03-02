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

def load_acq_img(filein):
    acq_img = np.load(filein)
    return acq_img



### Plot acquisition function 2d ONLY Random point - with overlay
def plot_acq_init(acq_file, file, evalfile):
    xax = np.arange(0, 0.5, 0.001)
    yax = np.arange(0, 0.5, 0.001)
    acq_img = load_acq_img(acq_file)

    # Acc data
    df = load_eval(evalfile)
    df_len = len(df)

    #init_points = df.iloc[:5]
    #iter_points = df.iloc[5:]

    plt.figure()
    plt.imshow(acq_img.T, origin='lower',extent=[xax[0],xax[-1],yax[0],yax[-1]])
    plt.colorbar()

    # Plot accuracy points
    sns.scatterplot(data=df, x="dropRate1", y="dropRate2", size="acc", 
                    sizes=(20, 50), color="red", marker="*", legend=False)
    # Text labels
    for i in range(1, df_len+1):
        plt.text(x=df['dropRate1'][i]+0.002, y=df['dropRate2'][i]+0.002, s=df['acc'][i], 
                fontdict=dict(color="black", size=10))
        
    plt.xlabel('dropRate1')
    plt.ylabel('dropRate2')
    plt.title('Acquisition function')
    plt.savefig(file)
    #plt.show()
    print("saved acq init")
    plt.close()

### Plot acquisition function 2d - with overlay
def plot_acq_full(acq_file, file, evalfile):
    xax = np.arange(0, 0.5, 0.001)
    yax = np.arange(0, 0.5, 0.001)
    acq_img = load_acq_img(acq_file)

    # Acc data
    df = load_eval(evalfile)
    df_len = len(df)

    init_points = df.iloc[:5]
    iter_points = df.iloc[5:]

    plt.figure()
    plt.imshow(acq_img.T, origin='lower',extent=[xax[0],xax[-1],yax[0],yax[-1]])
    plt.colorbar()

    # Plot accuracy points
    sns.scatterplot(data=init_points, x="dropRate1", y="dropRate2", size="acc", 
                    sizes=(10, 50), color="red", marker="*", legend=False)
    sns.scatterplot(data=iter_points, x="dropRate1", y="dropRate2", size="acc", 
                    sizes=(10, 50), color="black", marker="x", legend=False)

    # Text labels
    for i in range(1, df_len+1):
        plt.text(x=df['dropRate1'][i]+0.002, y=df['dropRate2'][i]+0.002, s=df['acc'][i], 
                fontdict=dict(color="black", size=10))
        
    # Arrows
    for i in range(6,df_len):
        x, y = iter_points['dropRate1'][i], iter_points['dropRate2'][i]
        dx, dy = (iter_points['dropRate1'][i+1] - x)*0.9, (iter_points['dropRate2'][i+1] - y)*0.9
        plt.arrow(x, y, dx, dy, 
                width = 0.0002, head_width = 0.005, linestyle='dotted')
        
    plt.xlabel('dropRate1')
    plt.ylabel('dropRate2')
    plt.title('Acquisition function')
    plt.savefig(file)
    #plt.show()
    print("saved acq full")
    plt.close()


### Plot performance over time
def plot_perf2(filein, random, fileto):
    df = load_eval(filein)
    rand = load_random(random)
    
    y_bo = np.maximum.accumulate(df['acc']).ravel()
    y_rand = np.maximum.accumulate(rand['acc']).ravel()
    max_len = max(len(y_bo), len(y_rand))
    xs = np.arange(1,max_len+1,1)
    plt.plot(xs, y_rand, 'o-', color = 'red', label='Random Search')
    plt.plot(xs, y_bo, 'o-', color = 'blue', label='Bayesian Optimization')
    plt.legend()
    plt.xlabel('Iteration')
    plt.ylabel('Best Accuracy')
    plt.title('Comparison between Random Search and Bayesian Optimization')
    plt.savefig(fileto)
    #plt.show()
    print("saved performance-over-time plot")
    plt.close()



if __name__ == "__main__":
    #plot_arrow("eval_file_4.txt", "arrow_plot.png")
    #plot_arrow5("eval_file_4.txt", "arrow_plot5.png")

    # plot_perf2("eval_file_4.txt", "randomSearchResults.npy",  "performance_plot.png")

    #plot_acq_rand()


    print()
    #return

    plot_acq_init("acq_img_init.npy", "plot_acq_init.png", "eval_file_iter.txt")
    plot_acq_full("acq_img_full.npy", "plot_acq_full.png", "eval_file_full.txt")
    plot_perf2("eval_file_4.txt", "randomSearchResults.npy",  "performance_plot.png")






