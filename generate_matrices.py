import os


import pandas as pd
import numpy as np
from vowpalwabbit.sklearn_vw import VWRegressor
from sklearn.linear_model import Lasso
from scipy.sparse import csr_matrix, eye, coo_matrix, save_npz
import math
from tqdm import tqdm
#from numba import jit

A_file = "./data/A.npz"
I_file = "./data/I.npz"
y_file = "./data/y.npz"

def convert_size(size, suffix):
   if size == 0:
       return "0B"
   size_name = ("", "K", "M", "G", "T", "P", "E", "Z", "Y")
   size_name = [s + suffix for s in size_name]
   i = int(math.floor(math.log(size, 1000)))
   p = math.pow(1000, i)
   s = round(size / p, 2)
   return "%s %s" % (s, size_name[i])

def convert_size_bytes(size_bytes):
   if size_bytes == 0:
       return "0B"
   size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
   i = int(math.floor(math.log(size_bytes, 1024)))
   p = math.pow(1024, i)
   s = round(size_bytes / p, 2)
   return "%s %s" % (s, size_name[i])

def get_density(mat):
    density = mat.getnnz() / np.prod(mat.shape)
    return density

def get_matrices(size):
    mult = 50

    cols = np.zeros(shape=size * mult, dtype=np.float32)
    rows = np.zeros(shape=size * mult, dtype=np.float32)
    values = np.zeros(shape=size * mult, dtype=np.float32)
    k = 0
    for i in tqdm(range(size)):
        # slow loop
        # r = np.zeros(shape = size)
        js = np.random.randint(size, size=mult)
        rs = np.random.random(size=mult)
        for n in range(0, mult):
            rows[k] = i
            cols[k] = js[n]
            values[k] = rs[n]
            k += 1
    return cols, rows, values


def get_random_matrices(size):
    y = np.ones(shape=size, dtype=np.float32)
    bytes_size_y = (y.size * y.itemsize)

    cols, rows, values = get_matrices(size)


    A = coo_matrix(coo_matrix((values, (rows, cols)), shape=(size, size))).tocsr()

    I = eye(y.shape[0], y.shape[0], dtype=np.float32, format="csr")


    bytes_size_A = A.data.nbytes + A.indptr.nbytes + A.indices.nbytes
    bytes_size_I = I.data.nbytes + I.indptr.nbytes + I.indices.nbytes

    print("A, I, y  = (%s, %s, %s)" % (convert_size_bytes(bytes_size_A), convert_size_bytes(bytes_size_I), convert_size_bytes(bytes_size_y)))
    print("A density", get_density(A))

    return A,I,y

if __name__ == "__main__":
    A,I,y = get_random_matrices(10**6)
    save_npz(A_file, A)
    save_npz(I_file, I)
    np.savez_compressed(y_file, y = y)




