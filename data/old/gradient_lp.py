import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np
import pandas as pd



#flows_df = pd.read_csv("data/flows.csv", index_col = "headings")

dep_df = pd.read_csv("data/depreciation.csv", index_col = "headings")

# 100 iron, 20 corn, 10 labour
# x0(1 iron + 10 corn  + 2 labour) ==> x0* bread
# x1(1 corn + 10 labour) ==> x1*30 corn
# argmax_x0{x0 * 1  + x0 * 10  + x0 * 2}
# argmax_x1{x1 * 1 + x1 * 10}
constraints = {}



def heq_zero(exp):
    return tf.nn.leaky_relu(exp, alpha = 10)
    #return tf.op.min(0,exp)

def leq_zero(exp):
    return tf.nn.leaky_relu(-exp, alpha = 10)

def eq_zero(exp):
    return tf.nn.leaky_relu(-exp) + tf.nn.leaky_relu(exp)

def tf_float(name):
    return tf.Variable(np.random.random() * 20, dtype=tf.float32, name = name)

def get_default_dict(dict, name):
    if(name not in dict):
        dict[name] = tf_float(name)

    return dict[name]


def process_flows_file(file_name):
    variables = {}
    multipliers = {}
    multipliers_debug = {}
    cost = None
    with open(file_name) as fp:
        for cnt, line in enumerate(fp):
            x_name = line.split()[-1]
            get_default_dict(variables, x_name)
            splitted = line.split("+")
            splitted[-1] = splitted[-1].split("-->")[0]
            splitted = [s.strip() for s in splitted]
            #print(splitted)
            for product in splitted:
                v_quantity, v_name = product.split()
                v_quantity = float(v_quantity.strip())
                v_name = v_name.strip()
                get_default_dict(variables, v_name)
                # if(cost is None):
                #     cost = v_quantity*get_default_dict(variables, v_name)
                # else:
                #     cost = cost + v_quantity*get_default_dict(variables, v_name)
                print(x_name, v_name, v_quantity)
                if(v_name in multipliers.keys()):
                    multipliers[v_name] = multipliers[v_name] + v_quantity*get_default_dict(variables, v_name)
                    multipliers_debug[v_name] = multipliers_debug[v_name] + [v_quantity ,v_name]

                else:
                    multipliers[v_name] = v_quantity*get_default_dict(variables, v_name)
                    multipliers_debug[v_name] = [v_quantity, v_name]
    cap_df = pd.read_csv("data/initial_stock.csv")
    print(multipliers, cost)
    cost = 0
    for output in cap_df.keys():
        #print(output)
        if (output in multipliers.keys()):

            final = float(cap_df[output][0])
            print(output, final)
            cost = cost   - (heq_zero(multipliers[output] - final))**2
    #print(multipliers_debug)
    #exit()

    # cost = 0
    # debug = []
    # for variable in list(variables.values()):
    #     final = float(cap_df[output][0])
    #     print(output, final)
    #     cost = cost - (variable-1)**2 #heq_zero(5-variable) #+ leq_zero(variable)
    #     debug +=[heq_zero(variable)] + [leq_zero(variable)]
    debug = []
    print(cost)

    #exit()
    #writer = tf.summary.FileWriter('./graphs', tf.get_default_graph())
    #exit()

    return cost, list(variables.values()), debug




cost, variables, debug = process_flows_file("./data/flows.flow")



print(tf.__version__)

goals = []




LR = 0.01
optm = tf.train.AdamOptimizer(LR).minimize(-cost)
init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

print(sess.run(variables))



for i in range(100000):
    if(i%1000 == 0 ):
        print(sess.run(variables), sess.run(cost), "crap")
        print([sess.run(d) for d in debug], "debug")
    (sess.run(optm))

sess.close()


