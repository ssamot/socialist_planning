import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("./plot_data/selviaria-noise.csv")

ax = sns.lineplot(x="tick",
                  y="$\mathcal{EG}_p$",
                  style="investment",
                  color = "gray",
                  data=df)

plt.legend(loc='upper right')
plt.savefig("./plots/selviaria-noise.pdf")
