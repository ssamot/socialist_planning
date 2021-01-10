import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
from utils import convert_size_round

df = pd.read_csv("plot_data/times.csv")
df.rename(columns={'population': 'profiles', 'household_goods': 'household goods'}, inplace=True)

df = df[df["household goods"] > 80]
columns = [ "industrial_goods"]

for column in columns:
    df[column] = np.log10(df[column])



grid = sns.FacetGrid(df, col="household goods",  hue="industrial_goods", palette="gray_r",
                      height=1.5, col_wrap=2)

grid.map(plt.plot, "bounded_time", "dependencies", marker="o")

grid.set_axis_labels("time in s", "dependencies")

household_goods =  [50, 100, 500,1000, 5000]
dependencies = [500,1000, 2000 ]

l_labels = [500,1000, 5000, 10000, 50000 ]
l_labels = [convert_size_round(k, " industrial goods") for k in l_labels]




plt.locator_params(axis='y', nbins=5)

yticks = plt.yticks()[0]
yticks = [convert_size_round(k, "") for k in dependencies]
grid.set(yticklabels = yticks )
plt.yticks(dependencies, labels=yticks)


fig = plt.gcf()
fig.set_size_inches(9, 6)




grid.fig.tight_layout(w_pad=1)

for ax in fig.axes:
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width , box.height*0.8])

legend = plt.legend(bbox_to_anchor=(-1.0, 0.9, 10, 1.99),loc="upper left",
      ncol=3)
for i in range(len(l_labels)):
   legend.get_texts()[i].set_text(l_labels[i])



plt.savefig("./plots/time.pdf")