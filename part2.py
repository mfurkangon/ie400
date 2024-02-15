"""
Group 39
Melih Rıza Yıldız
Mustafa Yetgin
Ömer Kağan Danacı
"""

import gurobipy as gp
from gurobipy import GRB

# Distances between nodes
distances = [
    [0.0, 1.0, 2.0, 1.0, 1.0, 2.0, 2.0, 1.0],
    [1.0, 0.0, 1.0, 1.0, 2.0, 3.0, 1.0, 3.0],
    [2.0, 1.0, 0.0, 1.0, 2.0, 2.0, 3.0, 2.0],
    [1.0, 1.0, 1.0, 0.0, 1.0, 2.0, 1.0, 2.0],
    [1.0, 2.0, 2.0, 1.0, 0.0, 1.0, 3.0, 2.0],
    [2.0, 3.0, 2.0, 2.0, 1.0, 0.0, 1.0, 2.0],
    [2.0, 1.0, 3.0, 1.0, 3.0, 1.0, 0.0, 1.0],
    [1.0, 3.0, 2.0, 2.0, 2.0, 2.0, 1.0, 0.0]
]

# Depot node distances for charging/refueling
depot_node_distances = {
    'X': [1, 1, 1, 2, 3, 2, 1, 1],
    'Y': [3, 2, 1, 1, 1, 1, 1, 2]
}
nodes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']

traveled_stations = [[(1, 9), (8, 16), (3, 11), (), (), (), (), (5, 13)],
                     [(4, 8, 12, 16), (1, 5, 9, 13), (), (), (), (), (2, 6, 10, 14), ()],
                     [(), (), (1, 6, 11), (5, 10, 15), (), (), (4, 9, 14), ()],
                     [(), (), (10,), (1,), (4,), (3,), (7,), ()],
                     [(), (), (4, 9, 14, 19), (), (1, 6, 11, 16), (2, 7, 12, 17), (), ()],
                     [(), (), (), (), (), (3, 7, 11, 15), (2, 6, 10, 14), (1, 5, 9, 13)],
                     [(3, 9), (), (), (), (), (3, 7, 11, 15), (2, 6, 10, 14), (1, 5, 9, 13)],
                     [(), (1, 7, 13), (6, 12, 18), (), (), (), (), (4, 10, 16)],
                     [(), (), (1, 7, 13), (), (3, 9, 15), (), (), (5, 11, 17)],
                     [(7,), (8,), (9,), (10,), (6, 11), (2,), (3,), ()],
                     [(1, 5, 9, 13, 17), (2, 6, 10, 14, 18), (3, 7, 11, 15, 19), (), (), (), (), ()],
                     [(1, 5, 9, 13, 17), (), (), (), (), (3, 7, 11, 15), (), ()],
                     [(), (2, 8, 14), (), (7, 13, 19), (), (5, 11, 17), (), ()],
                     [(), (), (4, 12), (), (6, 14), (), (1, 9), ()],
                     [(), (1, 4, 7, 10, 13, 16), (), (2, 5, 8, 11, 14, 17), (), (), (3, 6, 9, 12, 15, 18), ()]]

path_lengths = [17, 17, 16, 11, 20, 17, 15, 19, 19, 14, 20, 17, 20, 15, 19]


# Parameters
num_trains = 15
num_nodes = 8
max_electric_capacity = 8
max_diesel_capacity = 20
in_depot_electric_capacity = 3
in_depot_diesel_capacity = 2
on_route_charging_capacity = 1

# Costs
cost_in_depot_charging = 1000000
cost_on_route_charging = 350000
cost_in_depot_fuel = 800000
cost_electric_train = 750000
cost_diesel_train = 250000
cost_energy_diesel = 100000
cost_energy_electric = 20000

# Create a new model
model = gp.Model('TrainOptimization')

# Decision variables
y = {}
for i in range(1, num_trains + 1):
    for j in range(1, num_nodes + 1):
        for t in range(1, 21):
            y[i, j, t] = model.addVar(vtype=GRB.BINARY, name=f'y_{i}_{j}_{t}')

x = {}
for i in range(1, num_trains + 1):
    x[i] = model.addVar(vtype=GRB.BINARY, name=f'x_{i}')

z = {}
for j in range(1, num_nodes + 1):
    z[j] = model.addVar(vtype=GRB.INTEGER, name=f"z_{j}")

c = {}
for a in ['X', 'Y']:
    c[a] = model.addVar(vtype=GRB.INTEGER, name=f"c_{a}")
f = {}
for a in ['X', 'Y']:
    f[a] = model.addVar(vtype=GRB.INTEGER, name=f"f_{a}")

# Objective function: Minimize the total cost
purchasing_cost = cost_electric_train * gp.quicksum(
    1 - x[i] for i in range(1, num_trains + 1)) + cost_diesel_train * gp.quicksum(
    x[i] for i in range(1, num_trains + 1))

on_route_charging_cost = cost_on_route_charging * gp.quicksum(z[j] for j in range(1, num_nodes + 1))

in_depot_charging_fueling_cost = cost_in_depot_charging * gp.quicksum(
    c[a] for a in ['X', 'Y']) + cost_in_depot_fuel * gp.quicksum(f[a] for a in ['X', 'Y'])

travel_cost = cost_energy_diesel * gp.quicksum(
    x[i] * path_lengths[i - 1] for i in range(1, num_trains + 1)) + cost_energy_electric * gp.quicksum(
    (1 - x[i]) * path_lengths[i - 1] for i in range(1, num_trains + 1))

# Set the objective function to minimize the total cost
model.setObjective(purchasing_cost + on_route_charging_cost + in_depot_charging_fueling_cost + travel_cost,
                   GRB.MINIMIZE)

# Constraints
for j in range(1, num_nodes + 1):
    for t in range(1, 21):
        model.addConstr(gp.quicksum((1-x[i]) * y[i, j, t] for i in range(1, num_trains + 1)) <= z[j], f"con1_{j}_{t}")

for i in range(1, num_trains + 1):
    for s in range(1, 21):
        model.addConstr(
            gp.quicksum((x[i]*10 + y[i, j, t]) for j in range(1, num_nodes + 1) for t in range(1, s+1)) >= s // 8,
            f"con2_{i}")

model.addConstr(gp.quicksum(1 - x[i] for i in [1, 2, 6, 8, 10, 11, 12, 15]) <= c['X'] * 3)
model.addConstr(gp.quicksum(x[i] for i in [1, 2, 6, 8, 10, 11, 12, 15]) <= f['X'] * 2)
model.addConstr(gp.quicksum(1 - x[i] for i in [3, 4, 5, 7, 9, 13, 14]) <= c['Y'] * 3)
model.addConstr(gp.quicksum(x[i] for i in [3, 4, 5, 7, 9, 13, 14]) <= f['Y'] * 2)

for j in range(1, num_nodes + 1):
    model.addConstr(z[j] >= 0)

for a in ['X', 'Y']:
    model.addConstr(f[a] >= 0)

for a in ['X', 'Y']:
    model.addConstr(c[a] >= 0)

for i in range(1, num_trains + 1):
    for j in range(1, num_nodes + 1):
        valid_times = [time for time in traveled_stations[i - 1][j - 1] if time != ()]
        for t in range(1, 21):
            if t not in valid_times:
                model.addConstr(y[i, j, t] == 0)

# Optimize the model
model.optimize()

# Print the optimal solution
if model.status == GRB.OPTIMAL:
    print("Optimal solution found!")

    # Access and print the optimal values of x
    for i in range(1, num_trains + 1):
        if model.getVarByName(f'x_{i}').x == 0:
            print(f"Train_{i}: Electric")
        else:
            print(f"Train_{i}: Diesel")

    # Access and print the optimal values of z
    for j in range(1, num_nodes + 1):
        print(f"Number of Charging Stations for Node {nodes[j-1]}: {int(model.getVarByName(f'z_{j}').x)}")

    # Access and print the optimal values of c
    for a in ['X', 'Y']:
        print(f"Number of Charging Stations for Depot {a}: {int(model.getVarByName(f'c_{a}').x)}")

    # Access and print the optimal values of f
    for a in ['X', 'Y']:
        print(f"Number of Fueling Stations for Depot {a}: {int(model.getVarByName(f'f_{a}').x)}")

    # Print the objective value
    print(f"Total cost: ${model.objVal}")
else:
    print("No solution found.")