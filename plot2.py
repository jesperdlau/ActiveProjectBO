import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def load_eval(eval_file):
    # Import data
    df = pd.read_csv(eval_file, delimiter = "\t", index_col="Iteration")
    df.columns = ["acc", "dropRate1", "dropRate2"]
    df.index.name = "Iteration"
    df.index = df.index.astype('int8')
    #print(df)
    print("Loaded eval file")
    
    return df.round(2)
    


### Plot acquisition function 2d
def acqfunc(optimizer, file):
    xax = np.arange(0, 0.5, 0.01)
    yax = np.arange(0, 0.5, 0.01)
    pgrid = np.array(np.meshgrid(xax, yax,[1],[0],[1],[0],indexing='ij'))
    print(pgrid.reshape(6,-1).T.shape)
    #we then unfold the 4D array and simply pass it to the acqusition function
    acq_img = optimizer.acquisition.acquisition_function(pgrid.reshape(6,-1).T)
    #it is typical to scale this between 0 and 1:
    acq_img = (-acq_img - np.min(-acq_img))/(np.max(-acq_img - np.min(-acq_img)))
    #then fold it back into an image and plot
    acq_img = acq_img.reshape(pgrid[0].shape[:2])
    plt.figure()
    plt.imshow(acq_img.T, origin='lower',extent=[xax[0],xax[-1],yax[0],yax[-1]])
    plt.colorbar()
    plt.xlabel('dropRate1')
    plt.ylabel('dropRate2')
    plt.title('Acquisition function')
    plt.savefig(file)
    #plt.show()
    print("saved acq func file")
    plt.close()


### Plot performance over time
def plot_perf(opt, file):
    y_bo = np.maximum.accumulate(opt.Y).ravel()
    xs = np.arange(1,len(y_bo)+1,1)
    #plt.plot(xs, max_oob_per_iteration, 'o-', color = 'red', label='Random Search')
    plt.plot(xs, y_bo, 'o-', color = 'blue', label='Bayesian Optimization')
    plt.legend()
    plt.xlabel('Iterations')
    plt.ylabel('Error')
    plt.title('Comparison between Random Search and Bayesian Optimization')
    plt.savefig(file)
    #plt.show()
    print("saved performance-over-time plot")
    plt.close()


def plot_arrow(file, fileto):
    df = load_eval(file)
    df_len = len(df)

    init_points = df.iloc[:5]
    iter_points = df.iloc[5:]

    # Plot accuracy points
    sns.scatterplot(data=init_points, x="dropRate1", y="dropRate2", size="acc", 
                    sizes=(20, 200), color="red", marker="*", legend=False)
    sns.scatterplot(data=iter_points, x="dropRate1", y="dropRate2", size="acc", 
                    sizes=(20, 200), color="black", marker="o", legend=False)

    # Text labels
    for i in range(1, df_len+1):
        plt.text(x=df['dropRate1'][i]+0.002, y=df['dropRate2'][i]+0.002, s=df['acc'][i], 
                fontdict=dict(color="black", size=10))
        
    # Arrows
    for i in range(6,df_len):
        x, y = iter_points['dropRate1'][i], iter_points['dropRate2'][i]
        dx, dy = (iter_points['dropRate1'][i+1] - x)*0.9, (iter_points['dropRate2'][i+1] - y)*0.9
        plt.arrow(x, y, dx, dy, 
                width = 0.0002, head_width = 0.005)

    # Highlight optimum
    # for i in [12, 13, 14, 16]:
    #     x, y = results['droprate_1'][i], results['droprate_2'][i]
    #     plt.scatter(x = x, y = y,
    #                 color="blue", marker="^")

    # Kunne være interessant at se Baysian opt mål efter eks. 4, 9 og 10 iters.. 

    plt.xlim(0, 0.51)
    plt.ylim(0, 0.51)
    
    
    plt.savefig(fileto)
    print("saved arrow plot plot")
    #plt.show()

    print()
    plt.close()


if __name__ == "__main__":
    plot_arrow("eval_file_4.txt", "arrow_plot.png")

    plot_perf(opt, "performance_plot.png")


    #return




