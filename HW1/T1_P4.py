#####################
# CS 181, Spring 2022
# Homework 1, Problem 4
# Start Code
##################

import csv
import enum
import numpy as np
import matplotlib.pyplot as plt

csv_filename = 'data/year-sunspots-republicans.csv'
years  = []
republican_counts = []
sunspot_counts = []

with open(csv_filename, 'r') as csv_fh:

    # Parse as a CSV file.
    reader = csv.reader(csv_fh)

    # Skip the header line.
    next(reader, None)

    # Loop over the file.
    for row in reader:

        # Store the data.
        years.append(float(row[0]))
        sunspot_counts.append(float(row[1]))
        republican_counts.append(float(row[2]))

# Turn the data into numpy arrays.
years  = np.array(years)
republican_counts = np.array(republican_counts)
sunspot_counts = np.array(sunspot_counts)
last_year = 1985

# Plot the data.
plt.figure(1)
plt.plot(years, republican_counts, 'o')
plt.xlabel("Year")
plt.ylabel("Number of Republicans in Congress")
plt.figure(2)
plt.plot(years, sunspot_counts, 'o')
plt.xlabel("Year")
plt.ylabel("Number of Sunspots")
plt.figure(3)
plt.plot(sunspot_counts[years<last_year], republican_counts[years<last_year], 'o')
plt.xlabel("Number of Sunspots")
plt.ylabel("Number of Republicans in Congress")
plt.show()

# Create the simplest basis, with just the time and an offset.
X = np.vstack((np.ones(years.shape), years)).T

# TODO: basis functions
# Based on the letter input for part ('a','b','c','d'), output numpy arrays for the bases.
# The shape of arrays you return should be: (a) 24x6, (b) 24x12, (c) 24x6, (c) 24x26
# xx is the input of years (or any variable you want to turn into the appropriate basis).
# is_years is a Boolean variable which indicates whether or not the input variable is
# years; if so, is_years should be True, and if the input varible is sunspots, is_years
# should be false
def make_basis(xx,part='a',is_years=True):
#DO NOT CHANGE LINES 65-69
    if part == 'a' and is_years:
        xx = (xx - np.array([1960]*len(xx)))/40
        
    if part == "a" and not is_years:
        xx = xx/20

    phi_x = None
    if part == "a":
        phi_x = np.ones(shape=(len(xx), 6))
        for c, x in enumerate(xx):
            for a in range(1, 6):
                phi_x[c][a] = x ** a

    if part == "b":
        u_years = [1960, 1965, 1970, 1975, 1980, 1985, 1990, 1995, 2000, 2005, 2010]
        phi_x = np.ones(shape=(len(xx), 12))
        for c, x in enumerate(xx):
            for u in range(0, len(u_years)):
                phi_x[c][u] = np.exp(-((x - u_years[u]) ** 2) / 25)

    if part == "c":
        phi_x = np.ones(shape=(len(xx), 6))
        for c, x in enumerate(xx):
            for a in range(1, 6):
                phi_x[c][a] = np.cos(x / a)
    
    if part == "d":
        phi_x = np.ones(shape=(len(xx), 26))
        for c, x in enumerate(xx):
            for a in range(1, 26):
                phi_x[c][a] = np.cos(x / a)

    return phi_x

# Nothing fancy for outputs.
Y = republican_counts

# Find the regression weights using the Moore-Penrose pseudoinverse.
def find_weights(X,Y):
    w = np.dot(np.linalg.pinv(np.dot(X.T, X)), np.dot(X.T, Y))
    return w

# Compute the regression line on a grid of inputs.
# DO NOT CHANGE grid_years!!!!!
grid_years = np.linspace(1960, 2005, 200)
grid_sunspots = np.linspace(min(sunspot_counts), max(sunspot_counts), 200)

# TODO: plot and report sum of squared error for each basis
# Function to plot data and regression line for years and Republican count
def plot_rep_counts(letter):
    basis_years = make_basis(years, letter)
    w = find_weights(basis_years, republican_counts)
    X_basis_grid_years = make_basis(grid_years, letter)
    grid_Yhat = np.dot(X_basis_grid_years, w)

    # Calculate residual sum of squares error:
    y_hat = np.dot(basis_years, w)
    y_diff = 0

    for i in range(len(y_hat)):
        y_diff += ((y_hat[i] - republican_counts[i]) ** 2)

    plt.plot(years, republican_counts, 'o', label="data")
    plt.plot(grid_years, grid_Yhat, '-', label="Prediction")
    plt.xlabel("Years")
    plt.ylabel("Number of Republican in Congress")
    plt.title(f"Years v. Number of Republicans in Senate for letter {letter}")
    plt.savefig('Prob4-Years_v_Rep-' + letter + '.png')
    plt.legend()
    plt.show()

    return y_diff

# Function to plot data and regression line for sunspots and Republican count
def plot_sunspots(letter):
    basis_sunspots = make_basis(sunspot_counts[years<last_year], letter, is_years=False)
    w = find_weights(basis_sunspots, republican_counts[years<last_year])
    X_basis_grid_sunspots = make_basis(grid_sunspots, letter, is_years=False)
    grid_Y_hat = np.dot(X_basis_grid_sunspots, w)

    # Calculate residual sum of squares error:
    y_hat = np.dot(basis_sunspots, w)
    y_diff = 0

    for i in range(len(y_hat)):
        y_diff += ((y_hat[i] - republican_counts[years<last_year][i]) ** 2)

    plt.plot(sunspot_counts[years<last_year], republican_counts[years<last_year], 'o', label="data")
    plt.plot(grid_sunspots, grid_Y_hat, '-', label="Prediction")
    plt.xlabel("Number of Sunspots")
    plt.ylabel("Number of Republican in Congress")
    plt.title(f"Sunspots v. Number of Republicans in Senate for letter {letter}")
    plt.savefig('Prob4-Sunspots_v_Rep-' + letter + '.png')
    plt.legend()
    plt.show()

    return y_diff

# Plot the data and the regression line.
letters = ['a', 'b', 'c', 'd']
for i in letters:
    y_diff = plot_rep_counts(i)
    print(f"The residual sum of squares for Years v. Republican Counts is: {np.round(y_diff, decimals=4)} for letter {i}")

for i in letters:
    if i == 'b':
        continue
    y_diff = plot_sunspots(i)
    print(f"The residual sum of squares is for Sunspots v. Republican Counts is: {np.round(y_diff, decimals=4)} for letter {i}")