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
