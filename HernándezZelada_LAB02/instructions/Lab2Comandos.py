import numpy as np
import sympy as sp

x = np.array([1, 2, 3])         # Array
print("array x \n " + str(x))
y = np.arange(5)                # Array de 5 enteros contando el 0
print("array y \n " + str(y))
z = np.zeros((2, 3))            # Array de ceros de 2x3
print("array z \n " + str(z))

#Creacion de matrices con numpy
s = np.random.randint(10, size=(5, 5))   #matriz de 5x5 con valores enteros aleatorios con maximo de 10
print("matriz s: \n " + str(s))

a = np.array([[1, 2],                    #matriz con valores manuales
              [3, 4],
              [5, 6]])

#suma de matrices
print("matriz a \n " + str(a))
print ("suma a+a \n " + str(a+a))

#producto por un escalar
print("matriz a*3 \n " + str(a*3))
print("matriz -a  \n " + str(-a))



a = np.array([[1, 0], [2, -1]]) # Creacion de matriz con numpy
print("matriz a \n " + str(a))

#producto matricial
print("producto punto a*a \n " + str(np.dot(a, a)))

#punto elemento a elemento
print("producto elemento a elemento a*a \n " + str(a * a))


#producto cruz
u = np.arange(3)
v = 1 + u
print("u \n" + str(u) + " \nv\n " + str(v))
print("producto cruz " + str(np.cross(u, v)))


A = np.array([[1, 1, 0, 2, 0],
              [1, 1, 0, 3, 6],
              [1, 1, 1, 2, 3]])

print("A: \n " + str(A))
print("A transpuesta \n " + str(A.transpose()))         #Transpuesta de Matriz
print("FERR de A= \n " + str(sp.Matrix(A).rref()))      #FERR de una matriz
