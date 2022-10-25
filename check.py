import  numpy as np
import copy
time_list = [0, 0, 44, 44, 44, 44, 44, 44, 44, 44, 44, 76,
             79, 85, 115, 143, 164, 164, 162, 62, 36, 58,
             58, 58, 58, 166, 166, 166, 58, 166, 58, 166,
             58, 58, 166, 58, 173, 178, 43, 104, 252, 253,
             288, 345, 437]

time_list_copy = copy.deepcopy(time_list)

act_list = [0, 1, 57, 33, 21, 27, 51, 39, 45, 13, 3, 34,
            46, 40, 58, 52, 22, 28, 14, 4, 5, 29, 53, 35,
            6, 31, 55, 37, 47, 49, 41, 43, 23, 59, 61,
            16, 19, 25, 8, 10, 62, 63, 65, 64, 66]

sorted_act_list = np.zeros(len(act_list))
time_list.sort()
original_index0 = np.digitize(time_list_copy, time_list, right=True)
original_index1 = np.digitize(time_list_copy, time_list, right=False)
former_index = []
for i, index in enumerate(original_index0):
    unique_or_not = ~np.isin(index, former_index)
    if unique_or_not:
        sorted_act_list[index] = act_list[i]
        former_index.append(index)
    else:
        loc = np.where(index == former_index)[0].shape[0]
        sorted_act_list[index + loc] = act_list[i]
        former_index.append(index)

import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
state_matrix = np.array(pd.read_csv('stategraph.csv', header=0, index_col=0))
index = np.array([False, True,  True,  True, False,  True, False, False,  True,
        True, False, False,  True,  True, False, False, False, False,
       False, False,  True,  True, False, False, False, False,  True,
        True, False, False, False, False,  True,  True, False, False,
       False, False,  True,  True, False, False, False, False,  True,
        True, False, False, False, False,  True,  True, False, False,
       False, False,  True,  True, False, False, False, False, False,
       False, False, False, False])

selected_state_matrix = state_matrix
selected_state_matrix[~index, :][:, ~index] = 0
state_graph = nx.from_numpy_matrix(selected_state_matrix, create_using=nx.DiGraph)
nx.draw(state_graph, with_labels=True)
plt.show()


a = state_matrix[index]
b=a[:, index]
b_graph = nx.from_numpy_matrix(b, create_using=nx.DiGraph)
nx.draw(b_graph, with_labels=True)
plt.show()

