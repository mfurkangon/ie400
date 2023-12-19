import q1
import cplex

x_to_stations = q1.x_to_stations
y_to_stations = q1.y_to_stations

distances = q1.distances
paths = q1.paths

station = q1.station

model=cplex.Cplex()
# define the coefficients of the objective function
objective = [1e6, 8e5, 3.5e5, 7.5e5, 2.5e5, 2e4, 1e5]
variables= []

#Decision Variables

#Ti: Train type assigned to path i 0: diesel, 1: electical, i = 1,...,15
T = [f'T{i+1}' for i in range(15)]
T_types = ['B']*15
model.variables.add(names=T, types=T_types)

#Cj: Number of charge station assigned to node j, j=1,.....,8
C = [f'C{j+1}' for j in range(8)]
C_types = ['I']*8
C_lower_bound = [0]*8
model.variables.add(names=C, types=C_types, lb=C_lower_bound)

#Dk: Number of diesel fuel station in depot k, k = 1,2 k=1: Depot X, k=2: Depot Y
D = [f'D{j+1}' for j in range(2)]
D_types = ['I']*2
D_lower_bound = [0]*2
model.variables.add(names=D, types=D_types, lb=D_lower_bound)

#Ek: Number of charge station in depot k, k = 1,2 k=1: Depot X, k=2: Depot Y
E = [f'E{j+1}' for j in range(2)]
E_types = ['I']*2
E_lower_bound = [0]*2
model.variables.add(names=E, types=E_types, lb=E_lower_bound)

#Lijh: Train i is charged at node j at hour h, i = 1,....,15, j = 1,....,8, h = 1,....,20
L = [f'L{i+1}{j+1}{h+1}' for i in range(15) for j in range(8) for h in range(20)]
L_types = ['B']*15*8*20
model.variables.add(names=L, types=L_types)

#Fih: Is train i in depot at hour h, i = 1,....,15, h = 1,....,20
F = [f'F{i+1}{h+1}' for i in range(15) for h in range(20)]
F_types = ['B']*15*20
model.variables.add(names=F, types=F_types)

#Hih = How many hours passed since the last charge of train i at hour h
H = [f'H{i+1}{h+1}' for i in range(15) for h in range(20)]
H_types = ['I']*15*20
H_lower_bound = [0]*15*20
H_upper_bound = [8]*15*20
model.variables.add(names=H, types=H_types, lb=H_lower_bound, ub=H_upper_bound)

#Constraints
constraint_names = [f'Constraint{x+1}' for x in range(7)]

first = 'a'
rhs_first = 8
sense_first = 'L'
