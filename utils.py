import numpy as np

def full_matrix(production_df, demand_df, overcompletion):
    industries = list(production_df["Type"])
    indices = list(demand_df["Name"])

    print(production_df)
    print(demand_df)

    for idx in indices:
        production_df.insert(production_df.shape[1] - 1, idx, 0)

    for idx in indices:
        row = [0]*len(production_df.columns)
        #print(row)
        row[0] = idx
        row[-1] = overcompletion
        #print(row)
        k = dict(zip(production_df.columns, row))
        #print(k)

        production_df = production_df.append(k, ignore_index=True)

    basic_industries = list(demand_df.columns)[1:]
    print(basic_industries)

    for industry in basic_industries:
        for name in indices:
            value = float(demand_df.loc[demand_df["Name"] == name, industry])
            production_df.loc[production_df["Type"] == industry, name] = value



    print(production_df)
    matrix = production_df.values[:, 1:]
    return matrix


# def batch_generator(data, batch_size, replace=False, n_population = -1):
#     """Generate batches of data.
#     Given a list of numpy arrays, it iterates over the list and returns batches of the same size.
#     """
#     all_examples_indices = len(data[0])
#     while True:
#         mini_batch_indices = np.random.choice(all_examples_indices, size=batch_size, replace=replace)
#         tbr = [k[mini_batch_indices] for k in data]
#         yield tbr

def batch_generator_sparse(data, batch_size, replace=False, split = -1):
    """Generate batches of data.
    Given a list of numpy arrays, it iterates over the list and returns batches of the same size.
    """
    #goods_indices = range(0,n_population)
    #print(split, len(data))
    # print(data[2].shape)
    # print(data[2].sum())
    # exit()
    while True:
        mini_batch_indices_0 = np.random.randint(split, size=batch_size)
        #print(mini_batch_indices_0.shape)
        mini_batch_indices_1 = np.random.randint(split, data[0].shape[0], size=batch_size)
        #print(mini_batch_indices_1.shape)
        tbr = [(k[np.concatenate([mini_batch_indices_0, mini_batch_indices_1])]) for k in data]
        #print(tbr.shape)
        #exit()

        tbr[0] = tbr[0].toarray()
        tbr[1] = tbr[1].toarray()
        #print("dfsfd")
        yield tbr


# def batch_generator_sparse(data, batch_size, replace=False, split = -1):
#     """Generate batches of data.
#     Given a list of numpy arrays, it iterates over the list and returns batches of the same size.
#     """
#     #goods_indices = range(0,n_population)
#     #print(split, len(data))
#     # print(data[2].shape)
#     # print(data[2].sum())
#     # exit()
#     while True:
#         mini_batch_indices_0 = np.random.randint(data[0].shape[0], size=batch_size)
#         #print(mini_batch_indices_0.shape)
#         #mini_batch_indices_1 = np.random.randint(split, data[0].shape[0], size=batch_size)
#         #print(mini_batch_indices_1.shape)
#         tbr = [(k[mini_batch_indices_0]) for k in data]
#         #print(tbr.shape)
#         #exit()
#
#         tbr[0] = tbr[0].toarray()
#         tbr[1] = tbr[1].toarray()
#         #print("dfsfd")
#         yield tbr

def batch_generator(data, batch_size, replace=False, split = -1):
    """Generate batches of data.
    Given a list of numpy arrays, it iterates over the list and returns batches of the same size.
    """
    #goods_indices = range(0,n_population)
    #print(split, len(data))
    # print(data[2].shape)
    # print(data[2].sum())
    # exit()
    while True:
        mini_batch_indices_0 = np.random.randint(split, size=batch_size)
        mini_batch_indices_1 = np.random.randint(split, data[0].shape[0], size=batch_size)
        tbr = [(k[np.concatenate([mini_batch_indices_0, mini_batch_indices_1])]) for k in data]
        #print(tbr[2])
        #tbr[0] = tbr[0].todense()
        #tbr[1] = tbr[1].todense()
        #exit()
        #print(tbr[0].shape)

        #exit()
        yield tbr