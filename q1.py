import cplex

x_to_stations = [1, 1, 1, 2, 3, 2, 1, 1]
y_to_stations = [3, 2, 1, 1, 1, 1, 1, 2]

distances = [
                [0, 1, 2, 1, 1, 2, 2, 1], 
                [1, 0, 1, 1, 2, 3, 1, 3], 
                [2, 1, 0, 1, 2, 2, 3, 2], 
                [1, 1, 1, 0, 1, 2, 1, 2], 
                [1, 2, 2, 1, 0, 1, 3, 2],  
                [2, 3, 2, 2, 1, 0, 1, 2], 
                [2, 1, 3, 1, 3, 1, 0, 1], 
                [1, 3, 2, 2, 2, 2, 1, 0]
            ]

paths = [
            ['A', 'C', 'H', 'B'],
            ['B', 'G', 'A'],
            ['C', 'G', 'D'],
            ['D', 'F', 'E', 'G', 'C'],
            ['E', 'F', 'C'],
            ['H', 'G', 'F'],
            ['A', 'H', 'G', 'E'],
            ['B', 'H', 'C'],
            ['C', 'E', 'H'],
            ['F', 'G', 'E', 'A', 'B', 'C', 'D', 'E'],
            ['A', 'B', 'C'],
            ['A', 'F'],
            ['B', 'F', 'D'],
            ['G', 'C', 'E'],
            ['B', 'D', 'G']
        ]

stations = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7}

def calculate_path_length(path_index):
    path = paths[path_index]
    num_stations = len(path)

    path_length = 0

    for i in range(num_stations - 1):
        start = path[i]
        end = path[i + 1]
        
        path_length += distances[stations[start]][stations[end]]

    return path_length

def calculate_depot_distance(path_index, is_x):
    if is_x:
        dists = x_to_stations
    else:
        dists = y_to_stations

    path = paths[path_index]
    start = path[0]
    end = path[-1]

    return dists[stations[start]] + dists[stations[end]]

def calculate_loop_distance(path_index):
    path = paths[path_index]
    start = path[0]
    end = path[-1]

    return distances[stations[start]][stations[end]]


path_lengths = []
depot_distances = [[], []]
loop_distances = []

for i in range(15):
    is_x = True
    path_lengths.append(calculate_path_length(i))
    for j in range(2):
        depot_distances[j].append(calculate_depot_distance(i, is_x))
        is_x = False
    loop_distances.append(calculate_loop_distance(i))

# Create a CPLEX problem
problem = cplex.Cplex()

objective_coefficients = []
for i in range(1, 16):
    for j in range(15):
        # Coefficients for the terms involving x_i
        a = path_lengths[j]
        b1 = depot_distances[0][j]
        b2 = depot_distances[0][j]
        c = loop_distances[j]

        objective_coefficients.append((f"x{i}", a * f"z{j+1} + {b1} + {c} * (z{j+1} - 1)"))

        # Coefficients for the terms involving (x_i - 1)
        objective_coefficients.append((f"x{i}", a * f"z{j+1} + {b2} + {c} * (z{j+1} - 1)"))

problem.objective.set_linear(objective_coefficients)


# Set variable bounds
variable_names = [f"x{i}" for i in range(1, 16)] + [f"z{j+1}" for j in range(15)]
variable_types = ['B'] * 15 + ['I'] * 15
variable_lower_bounds = [0] * (15 + 15)
variable_upper_bounds = [1] * 15 + [cplex.infinity] * 15

problem.variables.add(names=variable_names, types=variable_types, lb=variable_lower_bounds, ub=variable_upper_bounds)

# Add individual constraints
for i in range(1, 16):
    constraint_coefficients = [(f"x{i}", a * f"z{j+1} + {depot_distances[j][0]} + {loop_distances[j]} * (z{j+1} - 1)") for j, a in enumerate(path_lengths)]
    problem.linear_constraints.add(lin_expr=[constraint_coefficients], senses=['L'], rhs=[20])

# Add global constraints
problem.linear_constraints.add(lin_expr=[[variable_names, [1] * 15]], senses=['L'], rhs=[10])
problem.linear_constraints.add(lin_expr=[[variable_names, [1] * 15]], senses=['G'], rhs=[5])

# Solve the problem
problem.solve()

# Get solution information
print("Objective Value:", problem.solution.get_objective_value())
print("Solution:", problem.solution.get_values())


