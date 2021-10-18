import random
import operator
import numpy as np

from deap import gp, creator, base, tools, algorithms
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from meta_tp3.datasource import get_dataframe

# PARAMETROS DO BENCHMARK
POP_SIZE = 100
TOURN_SIZE = 10
CROSSOVER_RATE = 0.9
MUTATION_RATE = 0.05
NUMBER_GEN = 10
MAX_TREE_DEPTH_INITIAL = 6
MAX_TREE_DEPTH_MUTATION = 15
TEST_SIZE = 0.25

# PARAMETROS GLOBAIS
MAX_TREE_DEPTH_INITIAL = 6
MAX_TREE_DEPTH_MUTATION = 15


# Definição do conjunto de dados
df = get_dataframe()
# A coluna de Churn, é nossa resposta, chamada aqui de Y
Y = df.pop("Churn")
# As demais são a entrada, chamada aqui de X
X = df.values


# Fazemos a divisão do tamanho de treinamento e teste
training_set, test_set, training_targets, test_targets = train_test_split(
    X, Y, test_size=TEST_SIZE
)


# Definição do indivíduo
pset = gp.PrimitiveSetTyped(
    "MAIN",
    [
        #  0   IDCliente               string
        str,
        #  1   Genero                  int64
        int,
        #  2   Aposentado              bool
        bool,
        #  3   Casado                  bool
        bool,
        #  4   Dependentes             object
        bool,
        #  5   MesesComoCliente        int64
        int,
        #  6   ServicoTelefone         bool
        bool,
        #  7   MultiplasLinhas         int64
        int,
        #  8   ServicoInternet         float64
        float,
        #  9   ServicoSegurancaOnline  int64
        int,
        #  10  ServicoBackupOnline     int64
        int,
        #  11  ProtecaoEquipamento     int64
        int,
        #  12  ServicoSuporteTecnico   int64
        int,
        #  13  ServicoStreamingTV      int64
        int,
        #  14  ServicoFilmes           int64
        int,
        #  15  TipoContrato            int64
        int,
        #  16  FaturaDigital           bool
        bool,
        #  17  FormaPagamento          int64
        int,
        #  18  ValorMensal             float64
        float,
        #  19  TotalGasto              float64
        float,
    ],
    bool,
)

# Funções para operadores
def protected_div(left, right):
    try:
        return left / right
    except ZeroDivisionError:
        return 1


def is_group(n, group):
    return n == group


def if_then_else(input, output1, output2):
    if input:
        return output1
    else:
        return output2


# Operadores
pset.addPrimitive(operator.ne, [bool, bool], bool)  # , name="!=")
pset.addPrimitive(operator.eq, [bool, bool], bool)  # , name="==")
pset.addPrimitive(operator.and_, [bool, bool], bool)  # , name="&&")
pset.addPrimitive(operator.or_, [bool, bool], bool)  # , name="||")
pset.addPrimitive(operator.not_, [bool], bool)  # , name="!")
pset.addPrimitive(if_then_else, [bool, bool, bool], bool)  # , name="if-then-else")

pset.addPrimitive(is_group, [int, int], bool)  # , name="ig")

pset.addPrimitive(operator.add, [float, float], float)  # , name="+")
pset.addPrimitive(operator.sub, [float, float], float)  # , name="-")
pset.addPrimitive(operator.mul, [float, float], float)  # , name="*")
pset.addPrimitive(protected_div, [float, float], float)  # , name="÷")
pset.addPrimitive(operator.lt, [float, float], bool)  # , name="<")
pset.addPrimitive(operator.gt, [float, float], bool)  # , name=">")

# Terminais
pset.addTerminal(False, bool, name="TRUE")
pset.addTerminal(True, bool, name="FALSE")
pset.addTerminal(1, int)
pset.addTerminal(2, int)
pset.addTerminal(3, int)
pset.addTerminal(4, int)

# Criadores - Individuos e Fitness
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)


# Toolbox
toolbox = base.Toolbox()


# Função de Avaliação - Retorna um valor de acurácia a partir da matriz de confusão
def fitness_function(individual, domain_set="TEST"):
    # Transforma a expressão de árvore em uma função chamável
    predict_is_churn = toolbox.compile(expr=individual)

    # Seleciona o conjunto e alvos para cálculo de fitness
    set_to_use = test_set if domain_set == "TEST" else training_set
    targets_to_use = test_targets if domain_set == "TEST" else training_targets

    # Faz as predições se os clientes do conjunto de treinamento é Churn ou não
    predictions = [predict_is_churn(*client) for client in set_to_use]

    tn, fp, fn, tp = confusion_matrix(predictions, targets_to_use).ravel()

    accuracy = (tp + tn) / (tp + tn + fp + fn)

    return (accuracy,)


toolbox.register(
    "expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=MAX_TREE_DEPTH_INITIAL
)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("evaluate", fitness_function)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("select", tools.selTournament, tournsize=TOURN_SIZE)
toolbox.register("compile", gp.compile, pset=pset)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genHalfAndHalf, min_=0, max_=MAX_TREE_DEPTH_MUTATION)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

# Decorators Bloat Control
toolbox.decorate(
    "mate",
    gp.staticLimit(
        key=operator.attrgetter("height"), max_value=MAX_TREE_DEPTH_MUTATION
    ),
)
toolbox.decorate(
    "mutate",
    gp.staticLimit(
        key=operator.attrgetter("height"), max_value=MAX_TREE_DEPTH_MUTATION
    ),
)


def main():
    pop = toolbox.population(n=POP_SIZE)

    hof = tools.HallOfFame(1)

    stats_fit = tools.Statistics(key=lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(key=len)

    stats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    algorithms.eaSimple(
        pop, toolbox, CROSSOVER_RATE, MUTATION_RATE, NUMBER_GEN, stats, halloffame=hof
    )

    return pop, stats, hof


if __name__ == "__main__":
    main()
