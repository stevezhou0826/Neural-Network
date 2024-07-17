import numpy as np

# verify the forward pass matrix multiplication works in linear layer
"""
# create a 4 by 6 np array with random integers
# between 0 and 9
alpha = np.random.randint(0, 10, (4, 6))
print("This is alpha\n")
print(alpha, "\n")

# create a 6 by 1 np array with all 1
x  = np.ones((6, 1))
print("This is x\n")
print(x, "\n")

# multiply alpha by x
beta = np.dot(alpha, x)
print("This is beta")
print(beta, "\n")
"""

# verify the backward pass matrix multiplication works in linear layer
"""
# create a 6 by 1 np array with random integers from 0 to 9
_x = np.random.randint(0, 10, (6, 1))
# trasnpose x and store in x_T
_x_T = _x.T
# create a 4 by 1 np array with random numbers from 0 to 9
_dlda = np.random.randint(0, 10, (4, 1))
# multiply _dlda by _x_T using np.outer
_dldx = np.outer(_dlda, _x_T)
# do the same thing with np.dot
_dldx2 = np.dot(_dlda, _x_T)

# print the results
print("This is _dlda\n")
print(_dlda, "\n")
print("This is _x_T\n")
print(_x_T, "\n")
print("This is _dldx\n")
print(_dldx, "\n")
print("This is _dldx2\n")
print(_dldx2, "\n")
"""

# test what will np.argmax return if there are multiple max values

# create a 3 by 1 np array with two 1s and one 0
test = np.array([[1], [1], [0]])
print("This is test\n")
print(test, "\n")
# find the max value in test
max = np.amax(test)
# find all the index of the max value in test
max_index = np.where(test == max)[0]
# find the index of the max value in test using where
max_index_where = np.where(test == max)
# print the results
print("This is max\n")
print(max, "\n")
print("This is max_index\n")
print(max_index, "\n")
print("This is max_index_where\n")
print(max_index_where, "\n")
