# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 19:15:52 2019
@author: danisnowman
"""
import numpy as np 
import sympy as sp
from matplotlib import pyplot as plt
import os

os.system('cls' if os.name == 'nt' else 'clear')
print("\n")

def Aumentar(a):
    Valor='h'+str(sp.Matrix(a).shape)
    Identidad=sp.eye(int(Valor[2]))
    for x in range(0, int(Valor[2])):
        Identidad=Identidad.col_insert(x, sp.Matrix(a).col(x))
    return Identidad

def InverseNum(matrix):
    try:
        invX = np.linalg.inv(matrix)
    except np.linalg.LinAlgError:
        print("No inverse found for Matrix:\n")
        invX = matrix
    return invX

def InverseSym(matrix):
    try:
        invX = sp.Matrix(matrix).inv()
    except :
        print("No inverse found for Matrix:\n")
        invX = matrix
    return invX

matrixA = np.array([[1,1,0],
                    [1,0,1],
                    [0,1,0]
                    ])

matrixB = np.array([[2,0,1],
                    [1,1,-4],
                    [3,7,-3]
                    ])

matrixC = np.array([[1,1,0],
                    [1,0,1],
                    [0,1,0],
                    [0,1,0]
                    ])

######### EJERCICIO 1 #########

print("\n\n---------------- 1 ----------------\n\n")
# SYMPY PRINTING
print("Pretty-printing with SymPy:\n")
print("Inverse of Matrix A:\n")
invA = InverseSym(matrixA)
sp.pprint(sp.Matrix(invA))
print("\n")

print("Inverse of Matrix B:\n")
invB = InverseSym(matrixB)
sp.pprint(sp.Matrix(invB))
print("\n")

print("Inverse of Matrix C:\n")
invC = InverseSym(matrixC)
sp.pprint(sp.Matrix(invC))
print("\n")

print("\n\n")


# NUMPY PRINTING
print("Printing with NumPy:\n")
print("Inverse of Matrix A:\n")
invA = InverseNum(matrixA)
print(invA,"\n")

print("Inverse of Matrix B:\n")
invB = InverseNum(matrixB)
print(invB,"\n")

print("Inverse of Matrix C:\n")
invC = InverseNum(matrixC)
print(invC,"\n")
print("\n\n")


# FERR 
print("Pretty-printing with SymPy but Inverting with RREF:\n")
print("Inverse of Matrix A:\n")
invA = Aumentar(matrixA).rref(simplify=True, pivots=False,normalize_last=True)
sp.pprint(invA)
print("\n")

######### EJERCICIO 2 #########        

print("\n\n---------------- 2 ----------------\n\n")

# prints rank and columns

# sets ranks
rangeA = np.linalg.matrix_rank(matrixA)
rangeB = np.linalg.matrix_rank(matrixB)
rangeC = np.linalg.matrix_rank(matrixC)

# sets columns
columnsA = matrixA.shape[1]
columnsB = matrixB.shape[1]
columnsC = matrixC.shape[1]

# sets nullspace
nullspaceA = sp.Matrix(matrixA).nullspace()
nullspaceB = sp.Matrix(matrixB).nullspace()
nullspaceC = sp.Matrix(matrixC).nullspace()

print("Matrix A\n Range: ", rangeA, "\n", "Columns: ", columnsA, "\n")
print("Matrix B\n Range: ", rangeB, "\n", "Columns: ", columnsB, "\n")
print("Matrix C\n Range: ", rangeC, "\n", "Columns: ", columnsC, "\n")

######### EJERCICIO 3 #########        

print("\n\n---------------- 3 ----------------\n\n")

# prints nullity and null space
print("Matrix A\n Nullity: ", rangeA - columnsA, "\n", "Null-space: \n")
sp.pprint(nullspaceA)
print("\n")

print("Matrix B\n Nullity: ", rangeB - columnsB, "\n", "Null-space: \n")
sp.pprint(nullspaceB)
print("\n")

print("Matrix C\n Nullity: ", rangeC - columnsC, "\n", "Null-space: \n")
sp.pprint(nullspaceC)
print("\n")


######### EJERCICIO 4 #########        

print("\n\n---------------- 4 ----------------\n\n")

def arrowed_spines(fig, ax):
    
    xmin, xmax = (-8,8)
    ymin, ymax = (-8,8)

    # get width and height of axes object to compute 
    # matching arrowhead length and width
    dps = fig.dpi_scale_trans.inverted()
    bbox = ax.get_window_extent().transformed(dps)
    width, height = bbox.width, bbox.height

    # manual arrowhead width and length
    hw = .5
    hl = .5
    lw = 1. # axis line width
    ohg = 0.3 # arrow overhang

    # compute matching arrowhead length and width
    yhw = hw/(ymax-ymin)*(xmax-xmin)* height/width 
    yhl = hl/(xmax-xmin)*(ymax-ymin)* width/height

    # draw x and y axis
    ax.arrow(xmin, 0, xmax-xmin, 0., fc='k', ec='k', lw = lw, 
             head_width=hw, head_length=hl, overhang = ohg, 
             length_includes_head= True, clip_on = True) 

    ax.arrow(0, ymin, 0., ymax-ymin, fc='k', ec='k', lw = lw, 
             head_width=yhw, head_length=yhl, overhang = ohg, 
             length_includes_head= True, clip_on = True)


# changing stylesheet
plt.style.use('seaborn-pastel')

# defining x range and functions
x = np.arange(-15,15,0.1)

fx = 2-x
gx = x-4
hx = x+4
ix = 6-x

min1 = np.minimum(ix, hx)
min2 = np.maximum(fx, gx)



plt.plot(x, fx, label="f(x)")
plt.plot(x, gx, label="g(x)")
plt.plot(x, hx, label="h(x)")
plt.plot(x, ix, label="i(x)")

plt.axhline(0, color="black")
plt.axvline(0, color="black")

# define corner points
x = [-1,3,5,1]
y = [3,-1,1,5]

# plot
l = plt.fill(x,y, facecolor=(.4,.9,.7,.3), alpha=0.6)
fig = plt.gcf()
ax = plt.gca()
arrowed_spines(fig, ax)


# limits
plt.xlim(-8, 8)
plt.ylim(-8, 8)


# add more ticks
ax.set_xticks(np.arange(-8,8))
ax.set_yticks(np.arange(-8,8))

# remove tick marks
ax.xaxis.set_tick_params(size=0)
ax.yaxis.set_tick_params(size=0)

# change the color of the top and right spines to opaque gray
ax.spines['right'].set_color((.8,.8,.8))
ax.spines['top'].set_color((.8,.8,.8))
# set title
plt.title("Bonus Graph:")

# set legend
plt.legend(loc="upper right")

# set gird
plt.grid('on',linestyle="--")
plt.xlabel("X Axis")
plt.ylabel("Y Axis")

# tweak the axis labels
xlab = ax.xaxis.get_label()
ylab = ax.yaxis.get_label()

xlab.set_style('italic')
xlab.set_size(10)
ylab.set_style('italic')
ylab.set_size(10)

# tweak the title
ttl = ax.title
ttl.set_weight('bold')

plt.savefig("bonus_graph.pdf")
plt.show()




