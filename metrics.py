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
        yield j, columns_civilians[i][j]




def humanity(A, I,  cit, n_goods,  model, sample = False):
    #print(cit, n_goods)
    columns_civilians = A[:, cit:]
    ones = np.ones(shape=(1, 3))

    h_min = np.inf
    for i in range(n_goods):
        #print(row_production)
        row_production = A[i]
        #row_production = row_production[0]
        #print(row_production.shape)
        A_input = np.array([row_production])
        I_input = np.array([I[i]])
        #print(A_input.shape, I_input.shape)
        #exit()
        y_hat = model.predict([A_input, I_input, ones])[0]
        per_civilian = y_hat/columns_civilians.shape[0]
        #for j,demand_civilian in enumerate(columns_civilians[i]):
        if(sample):
            s_func = sample
        else:
            s_func = full
        for j,demand_civilian in s_func(A,cit,i):

            if(demand_civilian == 0):
               continue
            expected = demand_civilian

            row_copy = row_production.copy()
            #print(row_copy)
            #exit()

            row_copy[cit+j] = 0
            A_input = np.array([row_copy])
            real = model.predict([A_input, I_input, ones])[0]
            score = (real/expected) + per_civilian
            #print(real, expected)
            h_min  = np.min([score,h_min])

    return h_min
            #exit()
            #print(row
            #_civilians)

