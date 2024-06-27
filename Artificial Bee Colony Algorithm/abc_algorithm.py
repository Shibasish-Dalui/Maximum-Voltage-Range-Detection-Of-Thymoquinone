import pandas as pd
import numpy as np
import os

# Load data

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
file_name = 'sample.xlsx'  # Change this to your file path
file_path = os.path.join(script_dir, '..', file_name)
data = pd.read_excel(file_path)
data = data.apply(pd.to_numeric, errors='coerce')
data = data.fillna(data.mean())


# Objective function

def objective_function(x):
    diffs = data_values - x
    return np.sum(diffs**2)


# Bee algorithm parameters
num_bees = 30
num_elite_sites = 5
num_best_sites = 10
elite_bees = 10
best_bees = 5
max_iterations = 100
search_radius = 0.1

# Initialize bee positions
data_values = data.values
bee_positions = np.random.uniform(
    data_values.min(), data_values.max(), (num_bees, data_values.shape[1]))

# Store best fitness values over iterations for plotting
best_fitness_values = []

# Main loop
for iteration in range(max_iterations):
    fitness = np.apply_along_axis(objective_function, 1, bee_positions)
    sorted_indices = np.argsort(fitness)
    bee_positions = bee_positions[sorted_indices]
    best_fitness = fitness[sorted_indices[0]]
    best_fitness_values.append(best_fitness)

    new_positions = []
    for i in range(num_elite_sites):
        for _ in range(elite_bees):
            new_position = bee_positions[i] + \
                np.random.uniform(-search_radius,
                                  search_radius, data_values.shape[1])
            new_positions.append(new_position)

    for i in range(num_elite_sites, num_best_sites):
        for _ in range(best_bees):
            new_position = bee_positions[i] + \
                np.random.uniform(-search_radius,
                                  search_radius, data_values.shape[1])
            new_positions.append(new_position)

    num_scouts = max(0, num_bees - len(new_positions))
    scout_positions = np.random.uniform(
        data_values.min(), data_values.max(), (num_scouts, data_values.shape[1]))
    new_positions.extend(scout_positions)

    bee_positions = np.array(new_positions)

best_bee = bee_positions[0]
final_best_fitness = objective_function(best_bee)

print("Best Position:", best_bee)
print("Best Fitness:", final_best_fitness)
