import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
from generate_matrices import convert_size_round

df = pd.read_csv("plot_data/times.csv")
df.rename(columns={'population': 'profiles', 'household_goods': 'household goods'}, inplace=True)

#df = df[df["population"] > 120]
df = df[df["household goods"] > 80]

columns = [ "industrial_goods"]

for column in columns:
    df[column] = np.log10(df[column])

print(df)
#exit()


grid = sns.FacetGrid(df, col="household goods", row = "profiles", hue="industrial_goods", palette="gray_r",
                      height=1.5)

grid.map(plt.plot, "bounded_time", "dependencies", marker="o")

grid.set_axis_labels("time in s", "dependencies")
#grid.set(yticklabels = [50, 100, 500,1000, 5000] )


#
household_goods =  [50, 100, 500,1000, 5000]
dependencies = [500,1000, 2000 ]

#print(grid)
#grid.despine(right=True)
#plt.legend(loc='upper right', mode = "expand")

l_labels = [500,1000, 5000, 10000, 50000 ]
l_labels = [convert_size_round(k, " industrial goods") for k in l_labels]




plt.locator_params(axis='y', nbins=5)

yticks = plt.yticks()[0]
print(yticks)
#
yticks = [convert_size_round(k, "") for k in dependencies]
print(yticks)
grid.set(yticklabels = yticks )
#
plt.yticks(dependencies, labels=yticks)





fig = plt.gcf()
fig.set_size_inches(18.5, 5.5)




grid.fig.tight_layout(w_pad=1)

for ax in fig.axes:
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width , box.height*0.8])

legend = plt.legend(bbox_to_anchor=(-3, 0.9, 2.6, 0.2),loc="upper left",
      ncol=5, mode="expand", borderaxespad=-2)
for i in range(len(l_labels)):
   legend.get_texts()[i].set_text(l_labels[i])



#grid.label(loc = 'upper right')


plt.savefig("./plots/time.pdf")