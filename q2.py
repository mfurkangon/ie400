import cplex

# values of the file depot_node_distances.txt
x_to_stations = [1, 1, 1, 2, 3, 2, 1, 1]
y_to_stations = [3, 2, 1, 1, 1, 1, 1, 2]

# values of the file distances.txt
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

# paths from the file paths.txt
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

# This dictionary helps convert the station name into its index
# Usage example: distance from station A to station F = distances[station['A'], station['F']]
station = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7}

def calculate_path_length(path_index):
    """ 
    Args:
        - path_index: index of the path from the paths array

    Returns:
        - length of the path, example: for A-C-F it will return (distance A to C) + (distance C to F)
    """

    path = paths[path_index]
    num_stations = len(path)

    path_length = 0

    for i in range(num_stations - 1):
        start = path[i]
        end = path[i + 1]
        
        path_length += distances[station[start]][station[end]]

    return path_length

def calculate_depot_distance(path_index, is_x):
    """ 
    Args:
        - path_index: index of the path from the paths array
        - is_x (Bool): x is used as the depot, if false it will consider that Y is the depot

    Returns:
        - The distance from the depot to the starting station of the path + the distance from the last station of the path to the depot
        - Example: for A-B-C if is_x == True it will return distance X to A + distance C to X
    """

    if is_x:
        dists = x_to_stations
    else:
        dists = y_to_stations

    path = paths[path_index]
    start = path[0]
    end = path[-1]

    return dists[station[start]] + dists[station[end]]

def calculate_loop_distance(path_index):
    """ 
    Args:
        - path_index: index of the path from the paths array

    Returns:
        - Distance between start station and end station of the path
    """


    path = paths[path_index]
    start = path[0]
    end = path[-1]

    return distances[station[start]][station[end]]


path_lengths = [0] * 15 # path lengths for all 15 paths
depot_distances = [[0] * 15, [0] * 15] # depot distances for all 15 paths (first sub array is for depot X and the second is for depot Y)
loop_distances = [0] * 15 # loop distances for all 15 paths

for i in range(15):
    is_x = True
    path_lengths[i] = calculate_path_length(i)
    for j in range(2):
        depot_distances[j][i] = calculate_depot_distance(i, is_x)
        is_x = False
    loop_distances[i] = calculate_loop_distance(i)

num_of_loops = [[0] * 15, [0] * 15] # The number of times we can loop through one path considering the restriction of 4h service per day (20h operation per day)
prev = 1
curr = 1
for i in range(15):
    for j in range(2):
        while (path_lengths[i] * curr + depot_distances[j][i] + loop_distances[i] * (curr - 1)) <= 20:
            prev = curr
            curr += 1
        num_of_loops[j][i] = prev