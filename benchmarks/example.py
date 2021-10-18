from meta_tp3.base import main
import pandas as pd

print("Executando Benchmark...")

NUMBER_OF_EXECUTIONS = 30

all_best_fitness = []

for i in range(NUMBER_OF_EXECUTIONS):
    pop, stats, hof = main()

    # Armazena o melhor indivíduo da execução
    best = hof.items[0]
    all_best_fitness.append(best.fitness.values[0])

print(all_best_fitness)
s = pd.Series(all_best_fitness)
print(s.describe())
