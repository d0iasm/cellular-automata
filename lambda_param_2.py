import matplotlib.pyplot as plt
import numpy as np
import random as rnd


def ca_1d(l, t, rule, cell_i, n, k):
    """
    Args:
    l: The number of cells
    t: The number of steps
    rule: Rule for transforming a next cell
    cell_i: initial cell array
    n: The number of status
    k: The number of neighboringstate
    Return: Cellular automata array
    """
    cell= cell_i
    data= [cell]
    for i in range(t):
        cell_next= [0 for i in range(l)]
        for j in range(l):
            neighboringstate= 0
            for m in range(k):
                neighboringstate += cell[int((j+(m-(k-1)/2)+l)%l)]*(n**(k-1-m))      
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
N = 2
K = 7
LAMBDA = 0.1
# SEED=100
# rnd.seed(SEED)

cell_init= [0 for i in range(L)]
cell_init[L//2]= 1

lambda_x = list(map(lambda x: x/100 , [i for i in range(110)]))
print(lambda_x)

fig = plt.figure(figsize=(5, 6))
ax = fig.add_subplot(1,1,1)
ax.set_xlim(-0.1, 1.0)
ax.set_ylim(-0.1, 1.0)
ax.set_xlabel("λ")
ax.set_ylabel("avg MI")
ax.set_title("λ and Mutual Information")
# ax.set()

# fig_mi = plt.figure(figsize=(5, 6))

for x in lambda_x:
    RULE= [(0 if rnd.random()<(1.0-x) else rnd.randint(1, N-1)) for i in range(N**K)]
    cell_data= ca_1d(L, T, RULE, cell_init, N, K)
    # ax.pcolor(np.array(cell_data), vmin = 0, vmax = N-1)
    mi_data = calc_ca_mi_list(cell_data)
    avg_mi = calc_ca_mi(mi_data)
    # print(x, avg_mi)
    ax.scatter(x, avg_mi, 7, 'k')
    if x % 0.1 == 0: print(x, avg_mi)

# plt.plot(0.4, 0.4, '.')
plt.show()
