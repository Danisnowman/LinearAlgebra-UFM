import numpy as nmp

# Creating 3 variables, one with a value
num1 = 8
num2 = 34
resultado = 0

# Adding the first and second number
resultado = num1 + num2
print("Addition: %s + %s = %s \n" % (num1, num2, resultado))

# Subtracting the first and second number
resultado = num1 - num2
print("Sub: %s - %s = %s \n" % (num1, num2, resultado))

# Multiplying the first and second number
resultado = num1 * num2
print("Mult: %s * %s = %s \n" % (num1, num2, resultado))

# Dividing the first and second number
resultado = num1 / num2
print("Div: %s / %s = %2.5f \n" % (num1, num2, resultado))

# Raising the first number to the second
resultado = num1 ** num2
print("Exp: %s ^ %s = %2.5e \n" % (num1, num2, resultado))

# Creating the first matrix
matrix = nmp.random.randint(1, 9, (5, 5), int)
print("Matrix 5x5:\n %s \n\n" % str(matrix))

# Modifying the fist matrix by adding a '1' diagonal
nmp.fill_diagonal(matrix, 1)
print("Modified matrix 5x5:\n %s \n\n" % str(matrix))