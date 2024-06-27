import numpy as np
import pandas as pd


def objective_function(x, data):
    diff = data - x
    return np.sum(diff ** 2)


def initialize_population(pop_size, dim, bounds):
    pop = np.random.uniform(bounds[0], bounds[1], (pop_size, dim))
    return pop


def update_position(pop, alpha, beta, delta, a, bounds):
    pop_size, dim = pop.shape
    new_pop = np.zeros((pop_size, dim))

    for i in range(pop_size):
        for j in range(dim):
            r1, r2 = np.random.rand(), np.random.rand()
            A1 = 2 * a * r1 - a
            C1 = 2 * r2

            D_alpha = np.abs(C1 * alpha[j] - pop[i, j])
            X1 = alpha[j] - A1 * D_alpha

            r1, r2 = np.random.rand(), np.random.rand()
            A2 = 2 * a * r1 - a
            C2 = 2 * r2

            D_beta = np.abs(C2 * beta[j] - pop[i, j])
            X2 = beta[j] - A2 * D_beta

            r1, r2 = np.random.rand(), np.random.rand()
            A3 = 2 * a * r1 - a
            C3 = 2 * r2

            D_delta = np.abs(C3 * delta[j] - pop[i, j])
            X3 = delta[j] - A3 * D_delta

            new_pop[i, j] = (X1 + X2 + X3) / 3

        new_pop[i, :] = np.clip(new_pop[i, :], bounds[0], bounds[1])

    return new_pop


def gwo(objective_function, bounds, dim, pop_size, max_iter, data):
    pop = initialize_population(pop_size, dim, bounds)
    alpha, beta, delta = np.zeros(dim), np.zeros(dim), np.zeros(dim)
    alpha_score, beta_score, delta_score = float(
        "inf"), float("inf"), float("inf")

    convergence_curve = np.zeros(max_iter)

    for t in range(max_iter):
        for i in range(pop_size):
            fitness = objective_function(pop[i, :], data)

            if fitness < alpha_score:
                delta_score = beta_score
                delta = beta.copy()

                beta_score = alpha_score
                beta = alpha.copy()

                alpha_score = fitness
                alpha = pop[i, :].copy()

            elif fitness < beta_score:
                delta_score = beta_score
                delta = beta.copy()

                beta_score = fitness
                beta = pop[i, :].copy()

            elif fitness < delta_score:
                delta_score = fitness
                delta = pop[i, :].copy()

        a = 2 - t * (2 / max_iter)

        pop = update_position(pop, alpha, beta, delta, a, bounds)
        convergence_curve[t] = alpha_score

        print(f"Iteration: {t+1}, Best Score: {alpha_score}")

    return alpha, alpha_score, convergence_curve


def load_data(filename, exclude_rows):
    df = pd.read_excel(filename)
    df = df.drop(exclude_rows)
    return df.values


# Main function
if __name__ == "__main__":
    filename = 'sample.xlsx'  # Replace with your actual file name
    exclude_rows = [0, 1]  # Replace with the indices of rows to exclude
    data = load_data(filename, exclude_rows)

    # Parameters
    bounds = [-10, 10]  # Adjust based on your problem
    dim = data.shape[1]
    pop_size = 20
    max_iter = 100

    best_solution, best_score, convergence_curve = gwo(
        objective_function, bounds, dim, pop_size, max_iter, data)

    print(f"Best Solution: {best_solution}")
    print(f"Best Score: {best_score}")
