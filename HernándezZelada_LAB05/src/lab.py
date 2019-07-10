"""
Created on Mon Jun 17 19:15:52 2019
@author: danisnowman
"""
import scipy as sc
import pandas as pd
import tkinter as tk
import numpy as np 
import sympy as sp
import tkinter.filedialog
import matplotlib
matplotlib.use('TkAgg')
# from tkinter import filedialog as fd
from scipy import optimize
from matplotlib import pyplot as plt
from sympy import pretty_print as pp
from sympy.abc import A,  T, b




""" 
 ######### EJERCICIO 1 #########

print("\n\n---------------- 1 ----------------\n\n")
# year_of_birth will be matrix A
year_of_birth = np.array([
    [1, 1920], 
    [1, 1930], 
    [1, 1940], 
    [1, 1950], 
    [1, 1960], 
    [1, 1970], 
    [1, 1980], 
    [1, 1990]])

# life_expectancy will be matrix b
life_expectancy = np.array([
    [54.1], 
    [59.7], 
    [62.9], 
    [68.2], 
    [69.7], 
    [70.8], 
    [73.7],
    [75.4]])

# Pretty-Printing A matrix and b matrix
print("Matrix A will be: \n")
pp(sp.Matrix(year_of_birth))

print("\n\nMatrix b will be: \n")
pp(sp.Matrix(life_expectancy))

# Normal Equations
equation_1 = (A**T * A)**T
equation_2 = (A**T * b)
At = year_of_birth.transpose()
AtA = np.dot(At, year_of_birth)
Atb = np.dot(At, life_expectancy)

# Pretty-Printing Normal Equations
print("\n")
pp(equation_1)
pp(sp.Matrix(AtA))
print("\n\n")
pp(equation_2)
pp(sp.Matrix(Atb))
print("\n")

# Normal Equation Solution
print("\nAT Ax = ATb \n")
x = np.dot(np.linalg.inv(AtA),Atb)
print("Lease Square Solutions: \n")
pp(sp.Matrix(x))


### Least Squares using np.linalg.lstsq ###

x = np.array([
    1920, 
    1930, 
    1940, 
    1950, 
    1960,
    1970, 
    1980, 
    1990
    ])

y = np.array([
    54.1, 
    59.7, 
    62.9, 
    68.2, 
    69.7, 
    70.8, 
    73.7, 
    75.4
    ])

A = np.vstack([x, np.ones(len(x))]).T
print("\nA:")
pp(sp.Matrix(A))
m, b = np.linalg.lstsq(A, y, rcond=None)[0]
print("\n\nThe Coefficients of the line are:")
print("Slope = ", round(m,4), "Intersect = ", round(b,4), "\n")
print("a. Prediction when x = 2000: ", round(((m*2000)+b),2))
print("b. Fairly good given the small margin of errors on each life expectancy in relation to the new line.")

graph1 = plt.figure(1)


# changing stylesheet
plt.style.use('seaborn-pastel')

# set gird
plt.grid('on',linestyle="--")
plt.xlabel("Years")
plt.ylabel("Life Expectancy")

plt.plot(x, y, "o",label="Real Facts")
plt.plot(x, m*x + b,label="Best Fitting Line")
plt.plot(x,y-m*x-b, "x", label="Errs.")
plt.plot(x,0*x,"--")
plt.title("Best Fitting Line").set_weight('bold')
plt.legend(loc="center right")
graph1.savefig("Graph_1.pdf")

plt.plot()

input("Press Enter to continue...\n\n") """



######### EJERCICIO 2 #########

print("\n\n---------------- 2 ----------------\n\n")


import_file_path = tk.filedialog.askopenfilename()
cities_data = pd.read_excel(import_file_path)
processed_data = cities_data.to_numpy()
x_processed_data = np.array(processed_data[:, [0]])
y_processed_data = np.array(processed_data[:, [1]])




A_2 = np.hstack([
    x_processed_data, 
    np.ones((x_processed_data.shape),dtype=float)
    ])
print("A =\n")
pp(sp.Matrix(A_2))
m_2, b_2 = np.linalg.lstsq(A_2, y_processed_data, rcond=None)[0]

m_2 = m_2[0]
b_2 = b_2[0]
print("\n\nThe Coefficients of the line are:")
print("Slope = ", round(m_2,4), "Intersect = ", round(b_2,4), "\n")
print("Maximized function: ",sc.maximum())


graph2 = plt.figure(2)

# changing stylesheet
plt.style.use('seaborn-pastel')

# set gird
plt.grid('on',linestyle="--")
plt.ylabel("Quantity")
plt.xlabel("Price")

plt.plot(x_processed_data, y_processed_data, "o",label="Real Facts")
plt.plot(x_processed_data, m_2*x_processed_data + b_2,label="Best Fitting Line")
plt.plot(x_processed_data, y_processed_data-m_2*x_processed_data-b_2, "x", label="Errs.")
plt.plot(x_processed_data, 0*x_processed_data,"--")
plt.title("Best Fitting Line").set_weight('bold')
plt.legend(loc="center right")
plt.savefig("Graph_2.pdf")

plt.plot()



plt.show()