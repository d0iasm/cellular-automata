import matplotlib.pyplot as plt
import numpy as np
# import random as rnd


def ca_1d(l, t, rule, cell_i):
    """
    Args:
        l: The number of cells
        t: The number of steps
        rule: Rule for transforming a next cell
        cell_i: initial cell array
    Return: Cellular automata array
    """
    cell= cell_i
    data= [cell]
    for i in range(t):
        cell_next= [0 for i in range(l)]
        for j in range(l):
            neighboringstate= cell[(j-1+l)%l]*4+cell[j]*2+cell[(j+1)%l]
            cell_next[j]= rule[neighboringstate]
        cell= cell_next
        data.append(cell)
    return(data)


def calc_entropy(data):
    """
    Args:
        data: An array
    Return: An entropy
    """
    dic = {}
    for d in data:
        dic[d] = dic[d] + 1 if d in dic else 1

    probdist = np.array(list(dic.values())) / len(data)
    return (np.sum([-p * np.log2(p) for p in probdist]))


def calc_joint_entropy(x, y):
    """
    Args:
        x: An array
        y: An array
    Return: The joint entropy of x and y 
    """
    xy = [(x[i], y[i]) for i in range(len(x))]
    return calc_entropy(xy)


def calc_mi(x, y):
    """
    Args:
        x: An array
        y: An array
    Return: The mutual information between x and y
    """
    return calc_entropy(x) + calc_entropy(y) - calc_joint_entropy(x, y)


def calc_ca_mi_list(data):
    """
    Args:
        data: A two dimensional array
    Return: The mutual information of each cell
    """
    t = len(data)
    l = len(data[0])
    mi_data = []
    for j in range(l): 
        data_j= [data[i][j] for i in range(t)]
        mi_data.append(calc_mi(data_j[:(t-1)], data_j[1:]))
    return mi_data


def calc_ca_mi(data):
    """
    Args:
    data: A two dimensional array
    Return: The average mutual information of an array
    """
    return sum(data) / len(data)



L = 101
T = 100
# SEED=100
# rnd.seed(SEED)

RNO = 100
RULE = [(RNO>>i) & 1 for i in range(8)]

#[0, 0, ..., 0, 1, 0, ..., 0, 0]
cell_init= [0 for i in range(L)]
cell_init[L//2]= 1

#random
# cell_init= [rnd.randint(0, 1) for i in range(L)]


cell_data= ca_1d(L, T, RULE, cell_init)
mi_data = calc_ca_mi_list(cell_data)
average_mi = calc_ca_mi(mi_data)

fig = plt.figure(figsize=(5, 6))
ax = fig.add_subplot(1,1,1)
ax.pcolor(np.array(cell_data), vmin = 0, vmax = 1,  cmap= plt.cm.binary)
ax.set_xlim(0, L)
ax.set_ylim(T-1, 0)
ax.set_xlabel("cell number")
ax.set_ylabel("step")
ax.set_title("rule" + str(RNO) + "MI=" + str(average_mi))

fig_mi = plt.figure(figsize=(5, 6))
x = [i for i in range(len(mi_data))]
plt.plot(x, mi_data)

plt.show()
