from q1 import * # no need to use q1. before vars and functions now
import cplex

model=cplex.Cplex()
#Decision Variables
"""
    WE NEED TWO MORE CONSTANS
    THOSE ARE:
    ## isAtNode(ijh) = THIS SHOWS RATHER THE TRAIN i is AT NODE j at the hour h (BINARY)
    ## timeToReachNode(ij) THIS  GIVES THE TIME TO REACH NEXT NODE FROM j FOR THE THE TRAIN i
    """
#Ti: Train type assigned to path i 0: diesel, 1: electical, i = 1,...,15
T = [f'T{i+1}' for i in range(15)]
T_types = ['B']*15
model.variables.add(names=T, types=T_types)

#Add to the objective function 
objective_coefficients = [750.0] * 15 + [250.0] * 15
variable_names = T + ['1-T{}'.format(i+1) for i in range(15)]


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

#THIS IS CONSTANT
#Fih: Is train i in depot at hour h, i = 1,....,15, h = 1,....,20
F = [f'F{i+1}{h+1}' for i in range(15) for h in range(20)]
F_types = ['B']*15*20
model.variables.add(names=F, types=F_types)

#Hih: How many hours passed since the last charge of train i at hour h
H = [f'H{i+1}{h+1}' for i in range(15) for h in range(20)]
H_types = ['I']*15*20
H_lower_bound = [0]*15*20
H_upper_bound = [8]*15*20
model.variables.add(names=H, types=H_types, lb=H_lower_bound, ub=H_upper_bound)

#THIS IS CCONSTANT
#Wi: Total working hour of train i
W = [f'W{i+1}' for i in range(15)]
W_types = ['I']*15
W_lower_bound = [0]*15
W_upper_bound = [20]*15
model.variables.add(names=W, types=W_types, lb=W_lower_bound, ub=W_upper_bound)

#Objective Function
#Objective Function
objective_function = {}
for i in range(15):
    if T[i] not in objective_function:
        objective_function[T[i]] = 0
    objective_function[T[i]] += 5e5

for k in range(2):
    if D[k] not in objective_function:
        objective_function[D[k]] = 0
    objective_function[D[k]] += 1e6
    if E[k] not in objective_function:
        objective_function[E[k]] = 0
    objective_function[E[k]] += 8e5

for j in range(8):
    if C[j] not in objective_function:
        objective_function[C[j]] = 0
    objective_function[C[j]] += 3.5e5

model.objective.set_sense(model.objective.sense.minimize)

model.objective.set_linear(objective_function.items())
model.objective.set_offset(2.5e4*15)
#Constraints
for i in range(15):
    # Sum(Hih) <= 8 for each i
    lin_expr = [[f'H{i+1}{h+1}' for h in range(20)], [1.0] * 20]
    model.linear_constraints.add(
        lin_expr=[lin_expr],
        senses=['L'],
        rhs=[8.0]
    )

print(model.solution.get_objective_value())

