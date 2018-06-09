import matplotlib.pyplot as plt
import numpy as np
import random as rnd


def ca_1d(l, t, rule, cell_i):
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
    # print(probdist)
    return (np.sum([-p * np.log2(p) for p in probdist]))

# 6.0: ７回中６回晴れという
# 7.0: 全施行回数
# 平均情報量 H(x) = -∑P(X=x)logP(X=x)
# return -6.0 / 7.0 + np.log2(6.0/7.0) - 1.0 / 7.0 * np.log2(1.0/7.0)


def calc_joint_entropy(x, y):
    """
    Args:
        x: An array
        y: An array
    Return: The joint entropy of x and y 
    """
    xy = [(x[i], y[i]) for i in range(len(x))]
    # print(xy)
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
    mi_data = []
    for i in range(len(data)-5):
        print(data[i])
        mi_data_one = [0 for i in range(len(data[0]))]
        for j in range(len(data[0])):
            x = [data[i+k][j] for k in range(5)]
            y = [data[i+k+1][j] for k in range(5)]
            mi_data_one[j] = calc_mi(x, y)
        mi_data.append(mi_data_one)
    return mi_data


def calc_ca_mi(data):
    """
    Args:
        data: A two dimensional array
    Return: The average mutual information of an array
    """
    t_data = list(zip(*data))
    return [sum(d) / len(d) for d in t_data]


print(calc_entropy([0, 1, 0, 1, 1]))
print(calc_entropy(['sunny', 'sunny', 'rain']))
print(calc_entropy([0, 1]))
print(calc_entropy([1,1,1,0,1,1,1]))
print(calc_entropy([1,2,0,0,1,0,2]))
print("---calc_joint_entropy")
print(calc_joint_entropy([1,1,1,0,1,1,1], [1,2,0,0,1,0,2]))

# Cell
ca = [1,0,1,1,0]
print(calc_joint_entropy(ca[0:-1], ca[1:]))

print("---calc_mi")

print(calc_mi([1,1,1,0,1,1,1], [1,2,0,0,1,0,2]))


L=101
T=100
# L = 5
# T = 10
# SEED=100
# rnd.seed(SEED)

RNO= 90
RULE= [(RNO>>i) & 1 for i in range(8)]

#[0, 0, ..., 0, 1, 0, ..., 0, 0]
cell_init= [0 for i in range(L)]
cell_init[L//2]= 1

#random
#cell_init= [rnd.randint(0, 1) for i in range(L)]


cell_data= ca_1d(L, T, RULE, cell_init)
print(cell_data)

print("---mi_data")

mi_data = calc_ca_mi_list(cell_data)
print(mi_data)

average_mi = calc_ca_mi(mi_data)
print(average_mi)

fig = plt.figure(figsize=(5, 6))
ax = fig.add_subplot(1,1,1)
ax.pcolor(np.array(cell_data), vmin = 0, vmax = 1,  cmap= plt.cm.binary)
ax.set_xlim(0, L)
ax.set_ylim(T-1, 0)
ax.set_xlabel("cell number")
ax.set_ylabel("step")
ax.set_title("rule" + str(RNO) + "MI=" + str(sum(average_mi)/len(average_mi)))

fig_mi = plt.figure(figsize=(5, 6))
x = [i for i in range(len(average_mi))]
plt.plot(x, average_mi)

plt.show()
