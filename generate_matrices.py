import numpy as np
from scipy.sparse import csr_matrix, eye, coo_matrix, save_npz
import math
from tqdm import tqdm

A_file = "./data/A.npz"
I_file = "./data/I.npz"
y_file = "./data/y.npz"

n_household_goods = 10**2
n_industrial_goods = 10**3
n_population = 10**4

def convert_size(size, suffix):
   if size == 0:
       return "0B"
   size_name = ("", "K", "M", "G", "T", "P", "E", "Z", "Y")
   size_name = [s + suffix for s in size_name]
   i = int(math.floor(math.log(size, 1000)))
   p = math.pow(1000, i)
   s = round(size / p, 2)
   return "%s%s" % (s, size_name[i])

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

def get_goods_matrices(size):
    mult = 50

    cols = np.zeros(shape=size * mult, dtype=np.float32)
    rows = np.zeros(shape=size * mult, dtype=np.float32)
    values = np.zeros(shape=size * mult, dtype=np.float32)
    k = 0
    for i in tqdm(range(size)):
        # slow loop
        # r = np.zeros(shape = size)
        js = np.random.randint(size, size=mult)
        rs = np.random.randint(10, size=mult)
        for n in range(0, mult):
            rows[k] = i
            cols[k] = js[n]
            values[k] = rs[n]
            k += 1
    return cols, rows, values


def get_matrices(n_industrial_goods, n_household_goods, n_population):
    mult = 50

    # cols = np.zeros(shape=size * mult, dtype=np.float32)
    # rows = np.zeros(shape=size * mult, dtype=np.float32)
    # values = np.zeros(shape=size * mult, dtype=np.float32)

    cols = []
    rows = []
    values = []

    #k = 0
    population_columns = list(range(n_industrial_goods + n_household_goods, n_population + n_industrial_goods + n_household_goods))
    #goods_columns = list(range(0 + n_industrial_goods + n_household_goods))

    #all_rows =  list(range(0, n_population + n_industrial_goods + n_household_goods))
    #l = list(np.random.randint(10, size=n_household_goods)) + [0] * (n_industrial_goods + n_population)
    #print(convert_size_bytes(n_population*n_household_goods * 24))
    #exit()
    for i in tqdm(range(n_household_goods)):
        cols.extend(population_columns)
        values.extend(np.random.randint(10, size=n_population))
        rows.extend([i]*n_population)

    print("finished goods")
    #print(convert_size_bytes(len(goods_columns) * len(goods_columns) * 32))
    cols_goods, rows_goods, values_goods = get_goods_matrices(n_industrial_goods + n_household_goods)

    cols.extend(cols_goods)
    rows.extend(rows_goods)
    values.extend((values_goods))
    return cols, rows, values


def get_random_matrices(n_industrial_goods, n_household_goods,  n_population):

    y_ones = np.ones(shape=n_population, dtype=np.float32)
    y_zeros = np.zeros(shape=n_household_goods + n_industrial_goods, dtype=np.float32)
    y = np.concatenate([y_zeros, y_ones])
    bytes_size_y = (y.size * y.itemsize)


    size = n_industrial_goods + n_household_goods + n_population
    cols, rows, values = get_matrices(n_industrial_goods, n_household_goods,  n_population)


    A = coo_matrix(coo_matrix((values, (rows, cols)), dtype = np.float32, shape=(size, size))).tocsr()

    I = eye(y.shape[0], y.shape[0], dtype=np.float32, format="csr")


    bytes_size_A = A.data.nbytes + A.indptr.nbytes + A.indices.nbytes
    bytes_size_I = I.data.nbytes + I.indptr.nbytes + I.indices.nbytes

    print("A, I, y  = (%s, %s, %s)" % (convert_size_bytes(bytes_size_A), convert_size_bytes(bytes_size_I), convert_size_bytes(bytes_size_y)))
    print("A density", get_density(A))

    return A,I,y

if __name__ == "__main__":
    print("Population of", convert_size(n_population, " People"))
    print("Industrial goods of size", convert_size(n_industrial_goods, ""))
    print("Household goods of size",  convert_size(n_household_goods, ""))
    A,I,y = get_random_matrices(n_industrial_goods, n_household_goods , n_population)
    save_npz(A_file, A)
    save_npz(I_file, I)
    np.savez_compressed(y_file, y = y)




