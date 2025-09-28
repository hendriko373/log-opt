from ortools.sat.python.cp_model import CpModel, CpSolver, OPTIMAL, FEASIBLE

import numpy as np


def find_shortest_tour(distances: np.ndarray, start: int) -> tuple[list[int], float]:
    N = distances.shape[0]
    cp_m = CpModel()

    x = []  # 1 if ij in tour
    d = []  # distance between ij
    u = []  # dummy var for ordering

    # Create variables
    for i in range(N):
        js = []
        ds = []
        for j in range(N):
            js.append(cp_m.NewBoolVar(f"x_{i}_{j}"))
            ds.append(distances[i, j])
        x.append(js)
        d.append(ds)
        u.append(cp_m.NewIntVar(1, N, f"u_{i}"))

    # Add constraints
    for i in range(N):
        cp_m.add(x[i][i] == 0)
        cp_m.add(sum(x[i][j] for j in range(N)) == 1)
    for j in range(N):
        cp_m.add(sum(x[i][j] for i in range(N)) == 1)
    for i in range(N):
        if i != start:
            for j in range(N):
                if j != start:
                    cp_m.add(u[i] - u[j] + x[i][j] * N <= (N - 1))
        else:
            cp_m.add(u[i] == 1)

    # Minimize cost
    cp_m.minimize(sum(d[i][j] * x[i][j] for i in range(N) for j in range(N)))

    solver = CpSolver()
    status = solver.solve(cp_m)

    result = [start]
    if status == OPTIMAL or status == FEASIBLE:
        while len(result) < N + 1:
            s = result[-1]
            es = [solver.value(x[s][j]) for j in range(N)]
            n = next(i for i, e in enumerate(es) if e == 1)
            result = result + [n]

    return result, solver.objective_value
