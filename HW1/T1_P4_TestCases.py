from cgi import test
import csv
import numpy as np
import matplotlib.pyplot as plt
import sys
from T1_P4 import make_basis, find_weights


'''
Instructions for using this Autograder:
1. Make sure it is in the same immediate directory/folder as your implementation file, which *must* be called T1_P4.py
2. Run this only after you have implemented the functions make_basis and find_weights for parts (a) and (d), or at least put in some dummy output for these parts so that the Autograder can run.
3. The test data used here switches the x and y axes from the original problem. The weights you see below are hence not the correct answer.
'''

test_x = np.array([36., 34., 32., 36., 43., 44., 42., 38., 38., 41., 53., 54., 53.,
       45., 45., 44., 43., 52., 55., 55., 50., 51., 55., 49.])
test_y = np.array([1960., 1962., 1964., 1966., 1968., 1970., 1972., 1974., 1976.,
       1978., 1980., 1982., 1984., 1986., 1988., 1990., 1992., 1994.,
       1996., 1998., 2000., 2002., 2004., 2006.])

w_a = [-4.89044882e+04,  6.10124740e+03, -2.89779054e+02,  6.81008355e+00,
       -7.91744489e-02,  3.64369239e-04]
w_d = [ 3.52546799e+05,  1.13711259e+00,  1.42221801e+02, -4.84584170e+03,
       -3.11976258e+04, -3.90732667e+04,  4.73383911e+05, -3.76335333e+06,
        1.01473722e+07, -6.95393244e+06, -1.21730618e+07,  1.14841006e+07,
        8.25016369e+06, -1.88335976e+06,  1.04782885e+06, -2.79499748e+06,
       -1.40391275e+07,  2.87475831e+06, -1.29943929e+06,  2.32150614e+06,
        8.35698821e+06, -1.01268476e+07,  1.12328526e+07,  1.01081055e+07,
        1.68244506e+06, -1.57402087e+07]
for part in ['a','d']:
    X = make_basis(test_x,part) 
    if part == 'a':
        if X.shape != (24,6):
            print("For part a, your X is the wrong shape. Shape should be (24,6) and you have ", X.shape)
    elif part == 'd':
        if X.shape != (24,26):
            print("For part d, your X is the wrong shape. Shape should be (24,26) and you have ", X.shape)
    w = find_weights(X,test_y)
    if part == 'a':
        if len(w_a)!=len(w):
            print("Your weights for part a have the wrong shape. Length should be ", len(w_a), " and you have ", len(w))
        else:
            parta_checker = np.array_equal(np.around(w_a,-1),np.around(w,-1))
            if parta_checker:
                parta_checker = "Pass"
            else:
                parta_checker = "Fail"
    elif part == 'd':
        if len(w_d)!=len(w):
            print("Your weights for part d have the wrong shape. Length should be ", len(w_d), " and you have ", len(w))
        else:
            partd_checker = np.array_equal(np.around(w_d,-1),np.around(w,-1))
            if partd_checker:
                partd_checker = "Pass"
            else:
                partd_checker = "Fail"
        
print("Your test case results for parts a and d respectively are:", parta_checker, partd_checker) 