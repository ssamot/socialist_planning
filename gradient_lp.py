import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np
import pandas as pd

# unused for the moment
dep_df = pd.read_csv("data/depreciation.csv", index_col="headings")

# 100 iron, 20 corn, 10 labour
# x0(1 iron + 10 corn  + 2 labour) ==> x0* bread
# x1(1 corn + 10 labour) ==> x1*30 corn
# argmax_x0{x0 * 1  + x0 * 10  + x0 * 2}
# argmax_x1{x1 * 1 + x1 * 10}
constraints = {}


def heq_zero(exp):
    return tf.nn.leaky_relu(exp, alpha=10)
    # return tf.op.min(0,exp)


def leq_zero(exp):
    return tf.nn.leaky_relu(-exp, alpha=10)


def eq_zero(exp):
    return tf.nn.leaky_relu(-exp) + tf.nn.leaky_relu(exp)


def tf_float(name):
    return tf.Variable((np.random.random() * -0.5) * 0.1, dtype=tf.float32, name=name)


def get_default_dict(dict, name):
    if (name not in dict):
        dict[name] = tf_float(name)
    return dict[name]


def process_flows_file(file_name):
    variables = {}
    multipliers = {}
    multipliers_debug = {}

    with open(file_name) as fp:
        for cnt, line in enumerate(fp):
            x_name = line.split()[-1]
            get_default_dict(variables, x_name)
            splitted = line.split("+")
            splitted[-1] = splitted[-1].split("-->")[0]
            splitted = [s.strip() for s in splitted]

            for product in splitted:
                v_quantity, v_name = product.split()
                v_quantity = float(v_quantity.strip())
                v_name = v_name.strip()
                get_default_dict(variables, v_name)

                print(x_name, v_name, v_quantity)
                if (v_name in multipliers.keys()):
                    multipliers[v_name] = multipliers[v_name] + v_quantity * get_default_dict(variables, v_name)
                    multipliers_debug[v_name] = multipliers_debug[v_name] + [v_quantity, v_name]

                else:
                    multipliers[v_name] = v_quantity * get_default_dict(variables, v_name)
                    multipliers_debug[v_name] = [v_quantity, v_name]
    cap_df = pd.read_csv("data/initial_stock.csv")
    cost = 0
    for output in cap_df.keys():
        if (output in multipliers.keys()):
            final = float(cap_df[output][0])
            #print(output, final)
            cost = cost - ((tf.abs(multipliers[output]) - final)) ** 2

    debug = []

    return cost, list(variables.values()), debug


def print_variables(file_name, variables):
    def rreplace(s, old, new):
        li = s.rsplit(old, 1)  # Split only once
        return new.join(li)

    with open(file_name) as fp:
        for cnt, line in enumerate(fp):
            x_name = line.split()[-1]
            splitted = line.split("+")
            k = splitted[-1].split("-->")

            splitted[-1] = k[0]
            splitted += [k[1]]

            new_line = []
            multiplier = variables[x_name]
            for sp in splitted:
                l = sp.strip().split(" ")
                new_line.append(str(float(l[0]) * abs(multiplier)) + " " + l[1])
                # new_line.append()
            new_line = [str(n) for n in new_line]
            new_line = " + ".join(new_line)
            new_line = rreplace(new_line, "+", "-->")
            print(new_line)

    print()


if __name__== "__main__":

    cost, variables, debug = process_flows_file("./data/flows.flow")

    LR = 0.01
    optm = tf.train.AdamOptimizer(LR).minimize(-cost)
    init = tf.global_variables_initializer()

    sess = tf.Session()
    sess.run(init)

    for i in range(100000):
        if (i % 1000 == 0):
            v_names = ([v.name.split(":")[0] for v in variables])
            v_values = sess.run(variables)

            cost_f = -sess.run(cost)
            print("Cost:", cost_f)
            print_variables("./data/flows.flow", dict(zip(v_names, v_values)))
            # if (cost_f < 0.0000000001):
            #     break
        (sess.run(optm))

    sess.close()
