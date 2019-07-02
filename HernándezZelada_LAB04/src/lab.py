import numpy as np
import sympy as sp 
import time
# import os

matrixA = np.array([[1,1,0],
                    [1,0,1],
                    [0,1,0]
                    ])

matrixB = np.array([[2,0,1,1],
                    [1,1,-4,0],
                    [3,7,-3,1]
                    ])

matrixC = np.array([[1,1,0,1],
                    [1,0,1,0],
                    [0,1,0,0],
                    [0,1,0,0]
                    ])

def det(matrix, SyOrNu):
    try:
        if SyOrNu == 1:
            symMatrix = sp.Matrix(matrix)
            detOutput = symMatrix.det()
        else: 
            detOutput = np.linalg.det(matrix)
    except:
        detOutput = "No determiant found because it's a non-square matrix."
    return detOutput

def diag(matrix):
    try:
        diag, D = sp.Matrix(matrix).diagonalize()
    except:
        diag = "It's non-diagonalizable"
    return diag

def charply(matrix):
    try:
        poly = sp.Matrix(matrix).charpoly().as_expr()
    except:
        poly = "It doesn't have a charpoly"
    return poly



######### EJERCICIO 1 #########

print("\n\n---------------- 1 ----------------\n\n")

# SYMPY PRINTING
print("Printing with SymPy:\n")
print("det() of Matrix A:\n")
syDetA = det(matrixA,1)
print(syDetA,"\n")


print("det() of Matrix B:\n")
syDetB = det(matrixB,1)
print(syDetB,"\n")


print("det() of Matrix C:\n")
syDetC = det(matrixC,1)
print(syDetC,"\n")
input("Press Enter to continue...\n\n")


# NUMPY PRINTING
print("\nPrinting with NumPy:\n")
print("det() of Matrix A:\n")
nuDetA = det(matrixA,2)
print(nuDetA,"\n")


print("det() of Matrix B:\n")
nuDetB = det(matrixB,2)
print(nuDetB,"\n")


print("det() of Matrix C:\n")
nuDetC = det(matrixC,2)
print(nuDetC,"\n")
input("Press Enter to continue...\n\n")

######### EJERCICIO 2 #########

print("\n\n---------------- 2 ----------------\n\n")

print("El output en sympy es: %s (entero) y el de numpy es %s (float)" % (type(syDetA), type(nuDetA)))
input("\nPress Enter to continue...\n\n")

######### EJERCICIO 3 #########

print("\n\n---------------- 3 ----------------\n\n")

# SYMPY PRINTING
print("Printing with SymPy:\n")

#### MATRIX A ####
print("Eigen Values of Matrix A:\n")
syEigenValuesStartTimeA = time.time()
syEigenValuesA = sp.Matrix(matrixA).eigenvals()
syEigenValuesDiffA = time.time() - syEigenValuesStartTimeA
sp.pprint(syEigenValuesA)
print("\nCalculated in %s seconds.\n" % (syEigenValuesDiffA))

print("Eigen Vectors of Matrix A:\n")
syEigenVectorsStartTimeA = time.time()
syEigenVectoresA = sp.Matrix(matrixA).eigenvects()
syEigenVectorsDiffA = time.time() - syEigenVectorsStartTimeA
sp.pprint(syEigenVectoresA)
print("\n")
print("\nCalculated in %s seconds.\n" % (syEigenVectorsDiffA))
input("Press Enter to continue...\n\n")



#### MATRIX C ####
print("Eigen Values of Matrix C:\n")
syEigenValuesStartTimeC = time.time()
syEigenValuesC = sp.Matrix(matrixC).eigenvals()
syEigenValuesDiffC = time.time() - syEigenValuesStartTimeC
print("\nCalculated in %s seconds.\n" % (syEigenValuesDiffC))
sp.pprint(syEigenValuesC)

print("Eigen Vectors of Matrix C:\n")
syEigenVectorsStartTimeC = time.time()
syEigenVectoresC = sp.Matrix(matrixC).eigenvects()
syEigenVectorsDiffC = time.time() - syEigenVectorsStartTimeC
sp.pprint(syEigenVectoresC)
print("\n")
print("\nCalculated in %s seconds.\n" % (syEigenVectorsDiffC))
input("Press Enter to continue...\n\n")

# NUMPY PRINTING
print("Printing with NumPy:\n")

#### MATRIX A ####
print("Eigen Values of Matrix A:\n")
nuEigenValuesStartTimeA = time.time()
nuEigenValuesA = np.linalg.eigvals(matrixA)
nuEigenValuesDiffA = time.time() - nuEigenValuesStartTimeA
print(nuEigenValuesA)
print("\nCalculated in %s seconds.\n" % (nuEigenValuesDiffA))

print("Eigen Vectors of Matrix A:\n")
nuEigenVectorsStartTimeA = time.time()
nuEigenValuesA, nuEigenVectoresA = np.linalg.eig(matrixA)
nuEigenVectorsDiffA = time.time() - nuEigenVectorsStartTimeA
print(nuEigenVectoresA)
print("\n")
print("\nCalculated in %s seconds.\n" % (nuEigenVectorsDiffA))
input("Press Enter to continue...\n\n")

#### MATRIX C ####
print("Eigen Values of Matrix C:\n")
nuEigenValuesStartTimeC = time.time()
nuEigenValuesC = np.linalg.eigvals(matrixC)
nuEigenValuesDiffC = time.time() - nuEigenValuesStartTimeC
print(nuEigenValuesC)
print("\nCalculated in %s seconds.\n" % (nuEigenValuesDiffC))

print("Eigen Vectors of Matrix C:\n")
nuEigenVectorsStartTimeC = time.time()
nuEigenValuesC, nuEigenVectoresC = np.linalg.eig(matrixC)
nuEigenVectorsDiffC = time.time() - nuEigenVectorsStartTimeC
print(nuEigenVectoresC)
print("\n")
print("\nCalculated in %s seconds.\n" % (nuEigenVectorsDiffC))
input("Press Enter to continue...\n\n")


######### EJERCICIO 4 #########
print("\n\n---------------- 4 ----------------\n\n")

### MATRIX A ###
print("Diagonalization of Matrix A:\n")
syDiagonalizationA = diag(matrixA)
sp.pprint(syDiagonalizationA)
print("\n Char Poly of Matrix A:")
syPolA = charply(matrixA)
sp.pprint(syPolA)
print("\n")
input("Press Enter to continue...\n\n")

### MATRIX C ###
print("Diagonalization of Matrix C:\n")
syDiagonalizationC = diag(matrixC)
sp.pprint(syDiagonalizationC)
print("\n Char Poly of Matrix C:")
syPolC = charply(matrixC)
sp.pprint(syPolC)
print("\n")
input("Press Enter to continue...\n\n")


print("\n\n---------------- BONO 1 ----------------\n\n")
print("Just Eigen Vectors of Matrix A:\n")
syEigenVectoresA = sp.Matrix(matrixA).eigenvects()
for i in range(0,2):
    print("Eigen Vector:\n")
    sp.pprint(syEigenVectoresA[i][2])
    print("\n\n")        

print("\n")
input("Press Enter to continue...\n\n")

print("Just Eigen Vectors of Matrix C:\n")
syEigenVectoresC = sp.Matrix(matrixC).eigenvects()
for i in range(0,3):
    print("Eigen Vector:\n")
    sp.pprint(syEigenVectoresC[i][2])
    print("\n\n")    

print("\n")
input("Press Enter to continue...\n\n")

print("\n\n---------------- BONO 2 ----------------\n\n")
syEigenValueAverageTime = (syEigenValuesDiffA + syEigenValuesDiffC)/2
syEigenVectorsAverageTime = (syEigenVectorsDiffA + syEigenVectorsDiffC)/2
nuEigenValueAverageTime = (nuEigenValuesDiffA + nuEigenValuesDiffC)/2
nuEigenVectorsAverageTime = (nuEigenVectorsDiffA + nuEigenVectorsDiffC)/2

print("Average time of Sympy's Eigen Values: %s" % syEigenValueAverageTime)
print("Average time of Numpy's Eigen Values: %s" % nuEigenValueAverageTime)
print("Average time of Sympy's Eigen Vectors: %s" % syEigenValueAverageTime)
print("Average time of Numpy's Eigen Vectors: %s" % nuEigenValueAverageTime)

perDiffVal = ((syEigenValueAverageTime - nuEigenValueAverageTime)/nuEigenValueAverageTime)
perDiffVec = ((syEigenVectorsAverageTime - nuEigenVectorsAverageTime)/nuEigenVectorsAverageTime)
print("Numpy is %s%% faster in values and %s%% faster in vectors" % (perDiffVal, perDiffVec)) 
