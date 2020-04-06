import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("./plot_data/butter.csv", index_col=0, delimiter=",")
#print(df.value)
#df.value = df.value.astype(float)


#exit()

#print(data)

ax = sns.lineplot(x="iteration", y="value", hue = "metric", data=df)
plt.savefig("./plot_data/butter.pdf")