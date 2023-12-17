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


path_lengths = [0] * 15
depot_distances = [[0] * 15, [0] * 15]
loop_distances = [0] * 15

for i in range(15):
    is_x = True
    path_lengths[i] = calculate_path_length(i)
    for j in range(2):
        depot_distances[j][i] = calculate_depot_distance(i, is_x)
        is_x = False
    loop_distances[i] = calculate_loop_distance(i)

num_of_loops = [[0] * 15, [0] * 15]
prev = 1
curr = 1
for i in range(15):
    for j in range(2):
        while (path_lengths[i] * curr + depot_distances[j][i] + loop_distances[i] * (curr - 1)) <= 20:
            prev = curr
            curr += 1
        num_of_loops[j][i] = prev


# Create a CPLEX problem
problem = cplex.Cplex()

x_names = [f"x{i+1}" for i in range(15)]
z_names = [f"z{i+1}" for i in range(15)]

# Add binary variables x
problem.variables.add(names=x_names, lb=[0]*len(x_names), ub=[1]*len(x_names), types=["B"]*len(x_names))

# Add integer variables z
problem.variables.add(names=z_names, lb=[0]*len(z_names), ub=[cplex.infinity]*len(z_names), types=["I"]*len(z_names))


problem.objective.set_sense(problem.objective.sense.minimize)

objective_function = {}
for i in range(15):
    # Summing up the coefficients for each x_names[i]
    if x_names[i] not in objective_function:
        objective_function[x_names[i]] = 0
    objective_function[x_names[i]] += depot_distances[0][i] - depot_distances[1][i]

# Add the offset
problem.objective.set_offset(sum(depot_distances[1]))

# set the linear part of the objective function
problem.objective.set_linear(objective_function.items())



# for simplicity
b = depot_distances

for i in range(15):

    rhs = 20
    coeff = b[0][i] - b[1][i]
    rhs = rhs - b[1][i]

    problem.linear_constraints.add(
        lin_expr=[[[x_names[i]], [coeff]]],
        senses=["L"],
        rhs=[rhs]
    )

problem.linear_constraints.add(
    lin_expr=[[x_names, [1] * len(x_names)]],
    senses=["L"],
    rhs=[10]
)

problem.linear_constraints.add(
    lin_expr=[[x_names, [1] * len(x_names)]],
    senses=["G"],
    rhs=[5]
)





try:
    # Solve the problem
    problem.solve()

    # Access the solution
    solution = problem.solution

    # Print solution status
    print("Solution status:", solution.get_status())

    # Print the solution
    solution = problem.solution
    for name in x_names:
        print(f"{name}: {solution.get_values(name)}")

    print("Objective value:", solution.get_objective_value())

except cplex.CplexError as e:
    print("Cplex Error:", e)