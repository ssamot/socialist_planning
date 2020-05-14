import numpy as np
from scipy.sparse import csr_matrix, eye, coo_matrix, save_npz, csc_matrix
import math
from tqdm import tqdm
import sys
from numba import jit


np.set_printoptions(precision=3, suppress=True)

A_file = "./data/A.npz"
I_file = "./data/I.npz"
y_file = "./data/y.npz"

n_household_goods = 5000
n_industrial_goods = 20000
n_population = 200
n_dependencies = 200

def convert_size(size, suffix):
   if size == 0:
       return "0" + suffix
   size_name = ("", "K", "M", "G", "T", "P", "E", "Z", "Y")
   size_name = [s + suffix for s in size_name]
   i = int(math.floor(math.log(size, 1000)))
   p = math.pow(1000, i)
   s = round(size / p, 2)
   return "%s%s" % (s, size_name[i])


def convert_size_round(size, suffix):
   if size == 0:
       return "0" + suffix
   size_name = ("", "K", "M", "G", "T", "P", "E", "Z", "Y")
   size_name = [s + suffix for s in size_name]
   i = int(math.floor(math.log(size, 1000)))
   p = math.pow(1000, i)
   s = int(round(size / p, 2))
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



def get_goods_matrices(size, mult):

    cols = []
    rows = []
    values = []

    for i in (range(size)):
        # slow loop
        # r = np.zeros(shape = size)
        cl_options = list(range(i,size))
        if(len(cl_options) == 0 ):
            continue
        m = mult
        #m = min(mult, len(cl_options))
        if(m > len(cl_options)):
            m = len(cl_options)
        js = np.random.choice(a = cl_options, size=m, replace=False)
        #print(js)
        #js = cl_options
        vs = np.random.random( size=m)*0.001
        #print(js)
        for n in range(0, m):
            rows.append(i)
            cols.append(js[n])
            values.append(vs[n])
            if(cols[-1] == rows[-1]):
                 values[-1] = 0.1
        #if(i%10000 == 0):
        #    print("Mem", convert_size_bytes(sys.getsizeof(values)), convert_size_bytes(sys.getsizeof(cols)), convert_size_bytes(sys.getsizeof(rows)))

    #cr = list(zip(cols,rows))



    return cols, rows, values


def get_matrices(n_industrial_goods, n_household_goods, n_population, dependencies):

    cols = []
    rows = []
    values = []


    #k = 0
    population_columns = list(range(n_industrial_goods + n_household_goods, n_population + n_industrial_goods + n_household_goods))

    for i in (range(n_household_goods)):
        #print(i, n_household_goods)
        cols.extend(population_columns)
        values.extend(np.random.random( size=n_population)*10)
        rows.extend([i]*n_population)


    print("house only", len(values), len(cols), len(rows))
    print("Finished citizen choices")
    #print(convert_size_bytes(len(goods_columns) * len(goods_columns) * 32))
    cols_goods, rows_goods, values_goods = get_goods_matrices(n_industrial_goods + n_household_goods, dependencies)

    cols.extend(cols_goods)
    rows.extend(rows_goods)
    values.extend((values_goods))

    values = np.array(values, dtype="float32")
    rows = np.array(rows, dtype="int32")
    cols = np.array(cols, dtype="int32")
    #print(cols)

    print("shapes", values.shape, rows.shape, cols.shape)
    print("Mem", convert_size_bytes(values.nbytes), convert_size_bytes(rows.nbytes), convert_size_bytes(cols.nbytes))


    return cols, rows, values


def get_random_matrices(n_industrial_goods, n_household_goods,  n_population, dependencies):

    #y_ones = np.ones(shape=n_population, dtype=np.float32)
    y_ones = np.random.random(size = n_population)
    y_ones = y_ones / np.sum(y_ones)
    #np.ones = np.random.dirichlet(np.ones(y_ones),size=1)
    y_zeros = np.zeros(shape=n_household_goods + n_industrial_goods, dtype=np.float32)
    y = np.concatenate([y_zeros, y_ones])
    bytes_size_y = (y.size * y.itemsize)


    size = n_industrial_goods + n_household_goods + n_population

    print(size, "size")
    cols, rows, values = get_matrices(n_industrial_goods, n_household_goods,  n_population, dependencies)
    #print(values)

    print("bfore matrix")

    A = coo_matrix((values, (rows, cols)), dtype = np.float32, shape=(size, size))
    print("after matrix")

    A = A.tocsc()
    #print(A.toarray())
    #exit()

    I = eye(y.shape[0], y.shape[0], dtype=np.float32, format="csc")


    bytes_size_A = A.data.nbytes + A.indptr.nbytes + A.indices.nbytes
    bytes_size_I = I.data.nbytes + I.indptr.nbytes + I.indices.nbytes

    print("A, I, y  = (%s, %s, %s)" % (convert_size_bytes(bytes_size_A), convert_size_bytes(bytes_size_I), convert_size_bytes(bytes_size_y)))
    print("A density", get_density(A))
    print("A shape", A.shape)


    #print(A.toarray())

    return A,I,y

#if __name__ == "__main__":
    print("Population of", convert_size(n_population, " People"))
    print("Industrial goods of size", convert_size(n_industrial_goods, ""))
    print("Household goods of size",  convert_size(n_household_goods, ""))
    A,I,y = get_random_matrices(n_industrial_goods, n_household_goods, n_population, n_dependencies)
    save_npz(A_file, A)
    save_npz(I_file, I)
    np.savez_compressed(y_file, y = y)


if __name__ == "__main__":
    filename = "./data/random_%s_%s_%s_%s_%s"
    #filename = "/mnt/backup/ssamot/data/random_%s_%s_%s_%s_%s"
    #for n_industrial_goods in [500, 1000, 5000, 10000, 50000]:
    #for n_industrial_goods in [100000]:
    for n_industrial_goods in [500, 1000, 5000, 10000, 50000]:
        for n_household_goods in [50, 100, 500, 1000, 5000]:
            for n_population in [150, 200]:
                for n_dependencies in [500, 1000, 2000, 3000, ]:
                    print(n_industrial_goods, n_household_goods, n_population, n_dependencies)
                    A, I, y = get_random_matrices(n_industrial_goods, n_household_goods, n_population, n_dependencies)

                    save_npz(filename%("A", n_industrial_goods, n_household_goods, n_population, n_dependencies), A)
                    save_npz(filename%("I", n_industrial_goods, n_household_goods, n_population, n_dependencies), I)
                    np.savez_compressed(filename%("y", n_industrial_goods, n_household_goods, n_population, n_dependencies), y=y)


