from q1 import * # no need to use q1. before vars and functions now
import cplex
import numpy as np

#Parameters
# calculate total working hours for each train
assigned_depots = problem.solution.get_values()

total_working_hours = [0]*15
for i in range(15):
    if assigned_depots[i]:
        total_working_hours[i] += x_to_stations[station[paths[i][0]]] + path_lengths[i]*num_of_loops[0][i] + x_to_stations[station[paths[i][-1]]]
    else:
        total_working_hours[i] += y_to_stations[station[paths[i][0]]] + path_lengths[i]*num_of_loops[0][i] + y_to_stations[station[paths[i][-1]]]

# create a 3D array showing is train i at node j  at hour h 
        #9th node = Depot X, 10th node = Depot Y

# add being in depot nodes
I = np.zeros((15, 10, 20), int)
for i in range(15):
    if assigned_depots[i]:
        I[i][8][0] = 1
    else:
        I[i][9][0] = 1

# add paths to the I array
for i in range(15):
    h = 0
    t = 0
    while t < num_of_loops[int(assigned_depots[i])][i]:
        for node in paths[i]:
            # from depot
            if not h:
                if assigned_depots[i]:
                    h += x_to_stations[station[node]]
                    prev = node
                else:
                    h += y_to_stations[station[node]]
                    prev = node
                I[i][station[node]][h]=1 
            # from an ordinary station
            else:
                h += distances[station[prev]][station[node]]
                prev = node
            I[i][station[node]][h]=1
        t += 1

print(I[0])


#create the cplex model and add the decision variables, objective function and the constraints.
model=cplex.Cplex()

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
D = [f'D{k+1}' for k in range(2)]
D_types = ['I']*2
D_lower_bound = [0]*2
model.variables.add(names=D, types=D_types, lb=D_lower_bound)

#Ek: Number of charge station in depot k, k = 1,2 k=1: Depot X, k=2: Depot Y
E = [f'E{k+1}' for k in range(2)]
E_types = ['I']*2
E_lower_bound = [0]*2
model.variables.add(names=E, types=E_types, lb=E_lower_bound)

#Lijh: Train i is charged at node j at hour h, i = 1,....,15, j = 1,....,8, h = 1,....,20
L = [f'L{i+1}{j+1}{h+1}' for i in range(15) for j in range(8) for h in range(20)]
L_types = ['B']*15*8*20
model.variables.add(names=L, types=L_types)

#Hih: How many hours passed since the last charge of train i at hour h
H = [f'H{i+1}{h+1}' for i in range(15) for h in range(20)]
H_types = ['I']*15*20
H_lower_bound = [0]*15*20
H_upper_bound = [8]*15*20
model.variables.add(names=H, types=H_types, lb=H_lower_bound, ub=H_upper_bound)

#Iijh: Is train i at node j at hour h, i = 1,....,15, j = 1,....,8, h = 1,....,20
I = [f'I{i+1}{j+1}{h+1}' for i in range(15) for j in range(8) for h in range(20)]
I_types = ['B']*15*8*20
model.variables.add(names=I, types=I_types)

## Rij: TIME TO REACH NEXT NODE FROM j FOR THE THE TRAIN i, i = 1,.....,15, j = 1,....,8
R = [f'R{i+1}{j+1}' for i in range(15) for j in range(8)]
R_types = ['I']*15*8
R_lower_bound = [0]*15*8
R_upper_bound = [3]*15*8 #maximum distance is 3 in distances.txt
model.variables.add(names=R, types=R_types, lb=R_lower_bound, ub=R_upper_bound)


#THIS IS CONSTANT
#Fih: Is train i in depot at hour h, i = 1,....,15, h = 1,....,20
#F = [f'F{i+1}{h+1}' for i in range(15) for h in range(20)]
#F_types = ['B']*15*20
#model.variables.add(names=F, types=F_types)

#THIS IS CCONSTANT
#Wi: Total working hour of train i
#W = [f'W{i+1}' for i in range(15)]
#W_types = ['I']*15
#W_lower_bound = [0]*15
#W_upper_bound = [20]*15
#model.variables.add(names=W, types=W_types, lb=W_lower_bound, ub=W_upper_bound)

#Objective Function
objective_function = {}
offset = 0
for k in range(2):
    if D[k] not in objective_function:
        objective_function[D[k]] = 0
    objective_function[D[k]] += 8e5
    if E[k] not in objective_function:
        objective_function[E[k]] = 0
    objective_function[E[k]] += 1e6

for j in range(8):
    if C[j] not in objective_function:
        objective_function[C[j]] = 0
    objective_function[C[j]] += 3.5e5

for i in range(15):
    if T[i] not in objective_function:
        objective_function[T[i]] = 0
    objective_function[T[i]] += 5e5 - total_working_hours[i]*8e4
    offset += 2.5e5+total_working_hours[i]*1e5

model.objective.set_sense(model.objective.sense.minimize)

model.objective.set_linear(objective_function.items())
model.objective.set_offset(offset)


#Constraints

for i in range(15):
    Ti = f'T{i+1}'
    Hih = [f'H{i+1}{h+1}' for h in range(20)]
    
    # Coefficients for Ti and Hih in the quadratic term
    quad_expr = [[Hih[j], 1, Ti, 1] for j in range(20)]
    
    # Coefficients for Ti and Hih in the linear term
    lin_expr = [[Hih + [Ti], [0] + [-1 for _ in range(20)]]]
    
    model.quadratic_constraints.add(
        quad_expr=quad_expr,
        lin_expr=lin_expr,
        sense='L',
        rhs=8
    )
print("problem?")
# Add Lijh <= Ti*Iijh
for i in range(15):
    Ti = f'T{i+1}'
    Lijh = [f'L{i+1}{j+1}{h+1}' for j in range(8) for h in range(20)]
    Iijh = [f'I{i+1}{j+1}{h+1}' for j in range(8) for h in range(20)]
    # Coefficients for Ti, Lijh, and Iijh in the quadratic term
    qmat = [[Lijh + [Ti] + Iijh, [1] + [-1] + [1 for _ in range(len(Iijh))]]]
    
    # Coefficients for Ti, Lijh, and Iijh in the linear term
    lin_expr = [[Lijh + [Ti] + Iijh, [1] + [0] + [0 for _ in range(len(Iijh))]]]
    
    model.quadratic_constraints.add(
        quad_expr=qmat,
        lin_expr=lin_expr,
        sense='L',
        rhs=0
    )
 
# Add Ti*Iijh*(Hih-Rij) <= 8
for i in range(15):
    Ti = f'T{i+1}'
    Iijh = [f'I{i+1}{j+1}{h+1}' for j in range(8) for h in range(20)]
    Hih = [f'H{i+1}{h+1}' for h in range(20)]
    Rij = [f'R{i+1}{j+1}' for j in range(8)]

    # Coefficients for the quadratic terms
    quad_expr = [[Ti, Iijh, Hih, Rij, 2, -1, -3]]  # Quadratic term: Ti * Iijh * (Hih - Rij)

    # Coefficients for the linear terms
    lin_expr = [[Ti, 1]]  # Linear term: Ti

    # Add the quadratic constraint to the model
    model.quadratic_constraints.add(
        quad_expr=quad_expr,
        lin_expr=lin_expr,
        sense='L',  # Less than or equal to
        rhs=8
    )

# Add Hih = Hi(h-1)*(1*Lijh)*(1-Dih) ???? bu ne la??????
   
# Add Cj = max(sum over i and h(Lijh)) for each j
for j in range(8):
    Cj = f'C{j+1}'
    Lijh = [f'L{i+1}{j+1}{h+1}' for i in range(15) for h in range(20)]

    # Coefficients for Cj and Lijh in the linear term
    #lin_expr = [[Cj + Lijh, [1] + [-1 for _ in range(len(Lijh))]]]
    lin_expr = [(Cj, 1)] + [(-1, Lijh[i][h]) for i in range(15) for h in range(20)]

    model.linear_constraints.add(
        lin_expr=lin_expr,
        senses=['E'],  # E for equality
        rhs=[0]
    )

no_electrical_trains_X = 0
no_electrical_trains_Y = 0
no_diesel_trains_X = 0
no_diesel_trains_Y = 0
for i in range(15):
    Ti = f'T{i+1}'
    if assigned_depots[i]:
        if Ti:
            no_electrical_trains_X += 1
        else:
            no_diesel_trains_X += 1
    else:
        if Ti:
            no_electrical_trains_Y += 1
        else:
            no_diesel_trains_Y += 1
        
# Add E0*3 >= no_electrical_trains_X
model.linear_constraints.add(
    lin_expr=[[['E0'], [3]]],
    senses=['G'],
    rhs=[no_electrical_trains_X]
)
        
# Add E1*3 >= no_electrical_trains_Y
model.linear_constraints.add(
    lin_expr=[[['E1'], [3]]],
    senses=['G'],
    rhs=[no_electrical_trains_Y]
)

# Add D0*2 >= no_diesel_trains_X
model.linear_constraints.add(
    lin_expr=[[['D0'], [2]]],
    senses=['G'],
    rhs=[no_diesel_trains_X]
)
        
# Add D1*2 >= no_diesel_trains_Y
model.linear_constraints.add(
    lin_expr=[[['D1'], [2]]],
    senses=['G'],
    rhs=[no_diesel_trains_Y]
)


print(model.solution.get_objective_value())

