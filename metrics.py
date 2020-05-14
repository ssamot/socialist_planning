import numpy as np

def calculate_percentages(X, x, production_df,  demand_df, model):
    people = list(demand_df["Name"])
    columns = list(demand_df.columns)[1:]
    #print(columns)
    perc = {}
    #print(production_df["Type"], indices)


    starting_position = production_df.shape[1] - len(columns)
    #print(production_df)
    #print(starting_position)
    completed_demand = demand_df.copy()
    l = []
    for i, id in enumerate(columns):
        for j, person in enumerate(people):
            r = production_df.loc[production_df["Type"] == id]
            #print(demand_df)
            expected = demand_df.loc[demand_df["Name"] == person][id]
            expected = expected.values[0]
            #print(expected)
            #exit()
            #print(demand_df)
            #print("==========")
           # print()
            X_row_copy = X[r.index.values[0]].copy()
            #X_copy[:,i] = 0
            # ones = np.ones(shape=(X.shape[0], 3))
            # l = model.predict([X_copy, ones])

            X_row_copy[starting_position + j] = 0

            h = (X_row_copy.dot(x[0]))/expected
            #print("num", X_row_copy, x[0])
            #print("num", )
            #print(id, person, X_row_copy.dot(x[0]), expected)
            completed_demand.loc[j,id] = h
            l.append(h)
    return np.array(l).min()
    #print(completed_demand)


def full(A, cit, i ):
    columns_civilians = A[:, cit:]
    for j, demand_civilian in enumerate(columns_civilians[i]):
        yield j, demand_civilian


def sample(A, cit, i, size = 100 ):
    columns_civilians = A[:, cit:]
    total_civilians = columns_civilians.shape[1]
    s = np.random.randint(total_civilians, size = size)
    #print(columns_civilians.shape, total_civilians)
    for j in s:
        #print(i,j)
        #print(columns_civilians[i,j])
        #exit()
        yield j, columns_civilians[i,j]


#
# there is obviously in insanaely faster way of doing this
def humanity(A, I, y,  cit, n_goods,  x, s = False, sparse = False):
    #print(cit, n_goods)
    #columns_civilians = A[:, cit:]


    h_min = np.inf
    for i in range(n_goods):
        if(sparse):
            row_production = A[i].toarray()
            A_input = row_production
            I_input = I[i].toarray()
        else:
            row_production = A[i]
            A_input = np.array([row_production])
            I_input = np.array([I[i]])
        if(s):
            s_func = sample
        else:
            s_func = full
        for j,demand_civilian in s_func(A,cit,i):

            if(demand_civilian == 0):
               continue

            A_input[0,cit+j] = 0
            real = (I_input - A_input ).dot(x)
            A_input[0,cit + j] = demand_civilian
            score = real/(demand_civilian*y[cit + j])
            h_min  = np.min([score,h_min])

    return np.float(h_min)


# def humanity(A, I, y, x, good_columns, profile_columns):
#     #for profile_column in profile_columns:
#         y_hat = (I[:,goods_columns]-A[:,profile_columns]).dot(x)





