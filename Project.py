import operator
import math
import random
import functools

import numpy as np
from deap import base, creator, tools, gp
import matplotlib.pyplot as plt

# توابع محافظت‌شده
def protectedDiv(left, right):
    try:
        return left / right
    except ZeroDivisionError:
        return 1

def protected_log(x):
    try:
        return math.log(abs(x))
    except ValueError:
        return 0

def protected_sqrt(x):
    try:
        return math.sqrt(abs(x))
    except ValueError:
        return 0

def protected_exp(x):
    try:
        return math.exp(x) if x < 700 else math.exp(700)
    except OverflowError:
        return math.exp(700)

# تعریف توابع پایه برای دو متغیر (x و y)
pset = gp.PrimitiveSet("MAIN", 2)
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(protectedDiv, 2)
pset.addPrimitive(math.sin, 1)
pset.addPrimitive(math.cos, 1)
pset.addPrimitive(protected_exp, 1)
pset.addPrimitive(protected_log, 1)
pset.addPrimitive(protected_sqrt, 1)
pset.addPrimitive(abs, 1)
pset.addEphemeralConstant("rand101", functools.partial(random.uniform, -1, 1))
pset.renameArguments(ARG0='x')
pset.renameArguments(ARG1='y')

# تنظیمات GP
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)
toolbox = base.Toolbox()
toolbox.register("expr_init", gp.genHalfAndHalf, pset=pset, min_=1, max_=3)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr_init)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# تابع هدف دو متغیره
def target_function(x, y):
    return x * math.sin(y) + y * math.cos(x)

# تابع شایستگی
def evalSymbReg(individual, points):
    func = toolbox.compile(expr=individual)
    sqerrors = [(func(x, y) - target_function(x, y))**2 for x, y in points]
    return math.fsum(sqerrors) / len(points),

toolbox.register("compile", gp.compile, pset=pset)
XY = [(random.uniform(-1,1), random.uniform(-1,1)) for _ in range(50)]
toolbox.register("evaluate", evalSymbReg, points=XY)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=5))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=5))

# اجرای الگوریتم GP
def main():
    random.seed(42)
    pop = toolbox.population(n=100)
    hof = tools.HallOfFame(1)
    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", np.mean)
    mstats.register("min", np.min)
    mstats.register("max", np.max)

    from deap import algorithms
    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2,
                                   ngen=40, stats=mstats,
                                   halloffame=hof, verbose=True)

    # نمودار MSE
    gen = log.select("gen")
    min_mse = log.chapters['fitness'].select('min')
    avg_mse = log.chapters['fitness'].select('avg')
    plt.figure()
    plt.plot(gen, min_mse, label='Min MSE')
    plt.plot(gen, avg_mse, label='Avg MSE')
    plt.xlabel('Generation')
    plt.ylabel('MSE')
    plt.title('MSE over Generations - 2D Function')
    plt.legend()
    plt.grid(True)
    plt.show()

    # نمایش بهترین فرد و دقت
    best = hof[0]
    print("Best individual:", best)

    real_vals = np.array([target_function(x, y) for x, y in XY])
    mse_final = evalSymbReg(best, XY)[0]
    accuracy = 100 * (1 - mse_final / np.var(real_vals))
    print(f"Accuracy: {accuracy:.2f}%")
    plt.figure()
    plt.bar(['Accuracy'], [accuracy])
    plt.ylim(0, 100)
    plt.ylabel('Accuracy (%)')
    plt.title('Model Accuracy - 2D Function')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()