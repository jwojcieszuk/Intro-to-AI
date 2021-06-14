# EARIN Project #1: Wojcieszuk Jakub, Gałecki Szymon
# Program implementing two optimization algorithms: Gradient Descent and Newton's method

import numpy as np
import time


# Function for checking corectness of a matrix input by a user
def check_matrix(M):

    # 1. determinant can't be equal to zero, matrix must be invertable
    if np.linalg.det(M) == 0:
        return False

    # 2. all eigenvalues must be bigger than zero
    if min(np.linalg.eig(M)[0]) <= 0:
        return False

    # 3. symmetric quadratic matrices are equal to their transposes
    if M.all() != M.T.all():
        return False

    # 4. if all condtions are satisfied, we have a positive-definite matrix
    return True



# Function for generating x0 by drawing numbers from a uniform distribution defined for the range [l, u].
def rand_x0(l, u, d):
    x0 = np.empty([d, 1])

    for i in range(d):
        x0[i] = np.random.uniform(low=l, high=u, size=None)

    return x0



# Batch mode
# Important: batch mode runs gradient descent with drawing random numbers from the desired range

# Why there is no Newton algorithm in batch mode: it is independent of starting point and will always deliver the same result - no need for iterating it
# --> Standard deviation for both minimum and function value will be equal to zero  
# --> Mean of minimum is minimum coordinates and mean of function is function value for minimum

# Why there is no option to provide starting point for gradient descent? Because each iteration would provide the same result --> no need for iterating it 

def batch_mode(n, A, b, c, d, l, u):

    # Arrays for storing results of each iteration of the batch mode
    G_x = np.zeros((d,n))
    G_Jx = np.array([])

    # Batch mode iteration loop
    for i in range(n):

        # Gradient descent
        J = lambda x: c + np.dot(b, x) + np.dot(np.dot(x.T, A), x)
        gradient = lambda x: 2*np.dot(A, x) + b.T

        # Gradient descent controlling variables
        iters = 0                 # iterations counter
        max_iter = 100000         # maximum number of iterations
        precision = 0.000001
        step_size = np.asarray([1])
        learning_rate = 0.01
        x_n = rand_x0(l, u, d)    # starting point
        x_n1 = x_n - learning_rate * gradient(x_n)
        max_time = 20
        comp_time = 0

        # Gradient descent iteration loop
        start_time = time.time()
        while max(step_size) > precision and iters < max_iter and comp_time < max_time:
            x_n = x_n1
            x_n1 = x_n - learning_rate * gradient(x_n)
            iters = iters + 1
            step_size = abs(x_n1 - x_n)
            comp_time = time.time() - start_time


        # Appending minimum coordinates and function value to the arrays
        for j in range(d):
            G_x[j][i] = np.round(x_n1[j], 2)
        
        G_Jx = np.append(G_Jx, J(x_n1))

    
    # Obtaining means and standard deviations after batch loop is finished
    Mx = []
    for i in range(d):
        Mx.append(np.mean(G_x[i]))

    MJx = np.mean(G_Jx)
    
    STDx = []
    for i in range(d):
        STDx.append(np.std(G_x[i]))

    STDJx = np.std(G_Jx)

    # Print the results
    print('Mean of x: ', Mx)
    print('Standard deviation of x: ', STDx)
    print('Mean of J(x): ', round(MJx, 2))
    print('Standard deviation of J(x): ', round(STDJx, 2))




# Gradient descent algorithm
def grad_descent(A, b, c, d, generate):

    if (generate == True):
        l = float(input("Enter lower bound of starting point range [l, u]:"))    
        u = float(input("Enter upper bound of starting point range [l, u]:"))
        x_n = rand_x0(l, u, d)  # initial vector
    if (generate == False):
        x_n = np.zeros((1, d))
        print("Enter starting point x0:")
        for i in range(d):
            x_n[0][i] = float(input())
        
        x_n = x_n.T

    # J(x) function definition
    J = lambda x: c + np.dot(b, x) + np.dot(np.dot(x.T, A), x)
    gradient = lambda x: 2*np.dot(A, x) + b.T

    i = 0  # iterations counter
    max_iter = 100000  # maximum number of iterations
    precision = 0.000001
    step_size = np.asarray([1])
    learning_rate = 0.01
    
    x_n1 = x_n - learning_rate * gradient(x_n)
    max_time = 20
    comp_time = 0

    print('\n\nPARAMETERS\n')
    print('A:\n', A, '\n\nb:\n', b, '\n\nc:\n', c,
          '\n\nd:\n', d, '\n\nx0:\n', x_n, '\n\n')

    print("Function value at x0:", J(x_n))

    start_time = time.time()
    while max(step_size) > precision and i < max_iter and comp_time < max_time:
        x_n = x_n1
        x_n1 = x_n - learning_rate * gradient(x_n)
        i = i + 1
        step_size = abs(x_n1 - x_n)
        comp_time = time.time() - start_time

    print("Number of iterations:", i, "\n")
    print("Minimum lies in:", np.round(x_n1.T, 2), "\n")
    print("Function value at minimum:", J(x_n1), "\n")
    print("Computation time:", round(comp_time * 1000, 2), "ms")



# Newton's algorithm
def newton(A, b, c, d, generate):

    # Drawing starting 
    if (generate == True):
        l = float(input("Enter lower bound of starting point range [l, u]:"))    
        u = float(input("Enter upper bound of starting point range [l, u]:"))
        x0 = rand_x0(l, u, d)  
    
    if (generate == False):
        x0 = np.zeros((1, d))
        print("Enter starting point x0:")
        for i in range(d):
            x0[0][i] = float(input())
        
        x0 = x0.T
 

    # Print out the parameters
    print('\n\nPARAMETERS\n')
    print('A:\n', A, '\n\nb:\n', b, '\n\nc:\n', c,
          '\n\nd:\n', d, '\n\nx0:\n', x0, '\n\n')

    # Funtion value at x0
    J = c + np.dot(b, x0) + np.dot(np.dot(x0.T, A), x0)
    print('J(x0) = ', J, '\n\n')

    # Algorithm
    x1 = x0 - np.dot(np.linalg.inv(A + A.T), b.T + np.dot((A + A.T), x0))
    print('Minimum - x1:\n', x1, '\n\n')

    # Funtion value at minimum, x1
    J1 = c + np.dot(b, x1) + np.dot(np.dot(x1.T, A), x1)
    print('Function value at minimum - J(x1) = ', J1, '\n\n')
    

# Function for obtaining parameters, choosing operation mode, and performing all user interactions
def run():

    print("Press \'enter\' after each number!")
    d = int(input("Enter dimension d:"))

    if d <= 0:
        print("Dimension has to be > 0")
        return

    b = np.zeros((1, d))
    A = np.zeros((d, d))

    c = float(input("Enter scalar number c:"))

    print("Enter vector b:")
    for i in range(d):
        b[0][i] = float(input())

    print("(Order: left to right, until the end of the row)")
    print("Enter matrix A:")
    for i in range(d):
        for j in range(d):
            A[i][j] = float(input())

    if (check_matrix(A) == False):
        print("Provided matrix does not satisfy conditions.")
        return

    correct_answer1 = False
    while(correct_answer1 != True):
        mode = input("Choose mode of operation. (s = single/b = batch) [s/b]: ")
        if (mode == 's'):
            starting_point = input("Choose starting point. (d = directly/ g = generate) [d/g]: ")
            if (starting_point == 'd'): 
                generate = False
            elif (starting_point == 'g'):
                generate = True
            else:
                print("Method of defining starting point was not selected, starting point will be generated.")
                generate = True

            correct_answer2 = False
            while(correct_answer2 != True):
                method = input("Choose optimization method. (g = gradient/n = newton) [g/n]: ")
                if (method == 'g'):
                    grad_descent(A, b, c, d, generate)
                    correct_answer2 = True
                    return
                if (method == 'n'):
                    newton(A, b, c, d, generate)
                    correct_answer2 = True
                    return

            correct_answer1 = True
            return

        if (mode == 'b'):
            n = int(input("Enter number of iterations n: "))
            if n <= 0:
                print("Number of iterations has to be > 0")
                return

            l = float(input("Enter lower bound of starting point range [l, u]: "))    
            u = float(input("Enter upper bound of starting point range [l, u]: "))

            batch_mode(n, A, b, c, d, l, u)

            correct_answer1 = True
            return    



run()
