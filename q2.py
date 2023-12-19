from q1 import * # no need to use q1. before vars and functions now
import cplex

x_to_stations = x_to_stations
y_to_stations = y_to_stations

distances = distances
paths = paths

station = station

model=cplex.Cplex()
# define the coefficients of the objective function
objective = [1e6, 8e5, 3.5e5, 7.5e5, 2.5e5, 2e4, 1e5]
variables= []

#decision variables

#Ti: Train type assigned to path i 0: diesel, 1: electical, i = 1,...,15
T = [f'T{i+1}' for i in range(15)]
T_types = ['B']*15

#Cj: Number of charge station assigned to node j, j=1,.....,8
C =[f'C{j+1}' for j in range(8)]
C_types = ['I']*8
C_lower_bound = [0]*8

#Dk: Number of diesel fuel station in depot k, k = 1,2 k=1: Depot X, k=2: Depot Y
D =[f'D{j+1}' for j in range(2)]
D_types = ['I']*2
D_lower_bound = [0]*2

#Ek: Number of charge station in depot k, k = 1,2 k=1: Depot X, k=2: Depot Y
E =[f'E{j+1}' for j in range(2)]
E_types = ['I']*2
E_lower_bound = [0]*2

#Lihj: 

