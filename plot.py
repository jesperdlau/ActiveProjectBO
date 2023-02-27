import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


results = np.array(
    [[0.98      , 0.1873    , 0.4754],
    [0.9762    , 0.366     , 0.2993],
    [0.9787    , 0.07801   , 0.078],
    [0.9812    , 0.02904   , 0.4331],
    [0.9737    , 0.3006    , 0.354],
    [0.9787    , 0.08323   , 0.5],
    [0.9762    , 0.07669   , 0.3734],
    [0.9825    , 0.05068   , 0.2608],
    [0.9825    , 0.002822  , 0.2319],
    [0.9775    , 0.3338    , 0.3211],
    [0.9775    , 0.01624   , 0.1076],
    [0.9812    , 0.2209    , 0.4122],
    [0.9837    , 0.04079   , 0.2237],
    [0.9837    , 0.08477   , 0.2205],
    [0.9837    , 0.3451    , 0.0612],
    [0.98      , 0.302     , 0.07584],
    [0.985     , 0.3906    , 0.04511],
    [0.9812    , 0.3637    , 0.001233],
    [0.9825    , 0.3957    , 0.09404],
    [0.9787    , 0.3893    , 0.04896]])

results = pd.DataFrame(results, columns=["accuracy", "droprate_1", "droprate_2"])
init_points = results[:5]
iter_points = results[5:]
print()

# Plot accuracy points
sns.scatterplot(data=init_points, x="droprate_1", y="droprate_2", size="accuracy", 
                sizes=(20, 200), color="red", marker="*", legend=False)
sns.scatterplot(data=iter_points, x="droprate_1", y="droprate_2", size="accuracy", 
                sizes=(20, 200), color="black", marker="o", legend=False)

# Text labels
for i in range(20):
    plt.text(x=results['droprate_1'][i]+0.002, y=results['droprate_2'][i]+0.002, s=results['accuracy'][i], 
             fontdict=dict(color="black", size=10))
    
# Arrows
for i in range(5,19):
    x, y = iter_points['droprate_1'][i], iter_points['droprate_2'][i]
    dx, dy = (iter_points['droprate_1'][i+1] - x)*0.9, (iter_points['droprate_2'][i+1] - y)*0.9
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
plt.show()

print()




