import math
import resource

import numpy as np


def full_matrix(production_df, demand_df, overcompletion):
    industries = list(production_df["Type"])
    indices = list(demand_df["Name"])

    for idx in indices:
        production_df.insert(production_df.shape[1] - 1, idx, 0)

    for idx in indices:
        row = [0]*len(production_df.columns)
        row[0] = idx
        row[-1] = overcompletion
        k = dict(zip(production_df.columns, row))


        production_df = production_df.append(k, ignore_index=True)

    basic_industries = list(demand_df.columns)[1:]


    for industry in basic_industries:
        for name in indices:
            value = float(demand_df.loc[demand_df["Name"] == name, industry])
            production_df.loc[production_df["Type"] == industry, name] = value



    matrix = production_df.values[:, 1:]
    return matrix, production_df


def memory_limit():
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    resource.setrlimit(resource.RLIMIT_AS, (get_memory() * 1024 / 2, hard))


def get_memory():
    with open('/proc/meminfo', 'r') as mem:
        free_memory = 0
        for i in mem:
            sline = i.split()
            if str(sline[0]) in ('MemFree:', 'Buffers:', 'Cached:'):
                free_memory += int(sline[1])
    return free_memory


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