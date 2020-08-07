import numpy as np
import cma

# Objective function
def objective_function(solutions):

    vector = np.zeros(32)
    probability = np.zeros(32)

    # Vector is filled with values
    vector = abs(np.sin(solutions))

    # Restrictions
    for idx,data in enumerate(vector):
        helper = data[0] + data[1]
        # When both values exceed 1
        if abs(solutions[idx][0]) > 1 or abs(solutions[idx][1]) > 1:
            probability[idx] = 100 + max(0,abs(solutions[idx][0]) - 1) + max(0,abs(solutions[idx][1]) - 1)

        # When no value exceeds 1
        else:
            probability[idx] = helper

    return probability

# Initialize optimizer
es = cma.CMAEvolutionStrategy(2*[2000],0.5,{'popsize':32})
i = 1

while i < 100:
    solutions = es.ask()
    es.tell(solutions, objective_function(solutions))
    es.disp()
    
    i  = i+1

a = es.result_pretty()
print(a.xbest)






















