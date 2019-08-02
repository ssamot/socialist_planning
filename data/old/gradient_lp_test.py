import tensorflow as tf
import numpy as np


## types of constraints

def heq_zero(exp):
    return tf.nn.leaky_relu(exp)

def leq_zero(exp):
    return tf.nn.leaky_relu(-exp)

def eq_zero(exp):
    return tf.nn.leaky_relu(-exp) + tf.nn.leaky_relu(exp)



print(tf.__version__)

goals = []
#
# x1 >= 1;
#       x2 >= 1;
#       x1 + x2 >= 2;



x1 = tf.Variable(np.random.random()*0.01, dtype = tf.float32)
x2 = tf.Variable(np.random.random()*0.01, dtype = tf.float32)


constraints = [leq_zero(x1 - 1), leq_zero(x2-1), leq_zero(x1+x2 - 2) ]

cost =  x1 + x2
for c in constraints:
    cost = cost + c

LR = 0.0001
optm = tf.train.AdamOptimizer(LR).minimize(cost)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

print(sess.run([x1,x2]))



for _ in range(10000):
    (sess.run(optm))

print(sess.run([x1,x2]))
sess.close()


