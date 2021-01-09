import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from utils import convert_size, convert_size_round
from scipy.interpolate import UnivariateSpline

VORPAL_PICK_ = "Units of Vorpal Pick +1"


def factory_1(order):
    return np.array([order*2, order*3])/1000

def factory_2(order):
    return np.array([np.sqrt(order)*10, order*5])/1000

def factory_3(order):
    return np.array([order*order/1000, order ])/1000


def generate_data():
    x = []
    y = []
    for order in range(1,2000):

            if (order > 500 and order <= 1000):
                y.append(factory_1(500) + factory_2(order-500))
                x.append(order)
            elif(order > 1000):
                #print(order)
                y.append(factory_1(500) + factory_2(500) + factory_3(order - 1000))
                x.append(order)
            else :
                y.append(factory_1(order))
                x.append(order)

    x = np.array(x)
    y = np.array(y)

    df = pd.DataFrame({"Lucloelium a":y.T[0]/x, "Lucloelium":y.T[0], "Vorpal Pick +1 a":y.T[1]/x, "Vorpal Pick +1":y.T[1]}, index = pd.Series(x, name=(
                "%s" % VORPAL_PICK_)))
    return df

def plot():
    df = generate_data()

    plt.figure(figsize=(6, 3))
    sns.lineplot(data=df[["Lucloelium a", "Vorpal Pick +1 a"]], palette="gray")
    xticks = plt.xticks()[0]


    xticks_tr = [convert_size(k, "") if k >= 0 else "" for k in xticks]
    xticks_tr[-1] = ""


    plt.xticks(xticks, labels=xticks_tr)
    plt.savefig("./plots/coeffs.pdf", bbox_inches='tight', pad_inches=0)
    plt.cla()

    plt.figure(figsize=(6, 3))

    sns.lineplot(data=df[["Lucloelium", "Vorpal Pick +1"]], palette="gray")

    xticks_tr = [convert_size(k, "") if k >= 0 else "" for k in xticks]
    xticks_tr[-1] = ""

    yticks = plt.yticks()[0]
    yticks_tr = [convert_size_round(k, "") if k >= 0 else "" for k in yticks]
    yticks_tr[-1] = ""

    plt.yticks(yticks, labels=yticks_tr)

    plt.savefig("./plots/units.pdf", bbox_inches='tight', pad_inches=0)
    plt.cla()


def get_io_approximations():
    df = generate_data()[["Lucloelium a", "Vorpal Pick +1 a"]]
    x = df.index

    y = df["Lucloelium a"].values
    a = UnivariateSpline(x, y, k = 1, s=0)

    y = df["Vorpal Pick +1 a"].values
    b = UnivariateSpline(x, y, k=1, s=0)

    return a, b


if __name__ == "__main__":
    plot()













