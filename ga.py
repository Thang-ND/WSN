from matplotlib import use
import matplotlib.pyplot as plt
import random
import copy 
import math
from itertools import product

H, W = (50, 50)
R = [5, 10]
UE = [x/2 for x in R] # UE = [2.5, 5]
cell_W, cell_H = (10, 10) # size of a cell
no_cells = W / cell_W * H / cell_H # number of cell in this AoI

targets = [] 

for h in range(int(abs(H/cell_H))):
    for w in range(int(abs(W/cell_W))):
        targets.append((w * cell_W + cell_W/2, h * cell_H + cell_H/2)) # target is located at center of each cell.

lower = [(ue, ue) for ue in UE] # lower x, lower y =  Rsi - UE
upper = [(H - l[0], W - l[1]) for l in lower]

min_noS = [int((H*W)/(36*ue*ue)) for ue in UE]
max_noS = [int((H*W)/(4*ue*ue)) for ue in UE]


class ObjectiveFunction(object):
    def __init__(self, targets, types=2, alpha1=1, alpha2=0, beta1=1, beta2=0.5, threshold=0.9):
        self.targets = targets
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.beta1 = beta1
        self.beta2 = beta2
        self.threshold = threshold
        self.sensor_types = range(types)
        self.diagonal = [self._distance([W, H], [R[i]+UE[i], R[i]+UE[i]]) for i in range(len(R))]

    def _distance(self, x, y):
        return math.sqrt((x[0] - y[0])**2 + (x[1] - y[1])**2)
    
    def _psm(self, x, y, type):
        distance = self._distance(x, y)

        if distance < R[type] - UE[type]:
            return 1
        if distance > R[type] + UE[type]:
            return 0
        else:
            lambda1 = UE[type] - R[type] + distance
            lambda2 = UE[type] + R[type] - distance
            lambda1 = math.pow(lambda1, self.beta1)
            lambda2 = math.pow(lambda2, self.beta2)
            return math.exp(-(self.alpha1 * lambda1/ lambda2 + self.alpha2))

    def get_fitness(self, x):
        used = [] # list contain sensors deployed in AoI
        for sensor in x:
            if sensor[0] < 0 or sensor[1] < 0:
                continue
            used.append(sensor)

        best_sol = float('-inf')
        best_trace = None
        best_covered = None
        best_no = None

        for case in product(self.sensor_types, repeat=len(used)):
            covered = []
            for t in targets:
                pt=1
                for index, sensor in enumerate(used):
                    p = self._psm(sensor, t, type=case[index])
                    if p == 0:
                        continue
                    pt = pt * p

                pt = 1 - pt

                if pt >= self.threshold:
                    covered.append(t)

            min_dist_sensor = float('+inf')
            for ia, a in enumerate(used):
                for ib, b in enumerate(used):
                    if a != b:
                        min_dist_sensor = min(min_dist_sensor, self._distance(a, b)/(R[case[ia]]*R[case[ib]]))

            ## 3s/3 objective 
            obj = (math.pow(len(covered), 10)/no_cells ) * (max(max_noS) - min(min_noS) + 1)/(len(used) - min(min_noS) + 1) * min_dist_sensor/max(self.diagonal)
            # ---- NEff/ number of cells---------------------- 1/ SensorCost----------------------- MD()
            
            if obj > best_sol:
                best_sol = obj
                best_trace = case
                best_covered = len(covered)
                best_no = len(used)
                #return best_sol, (best_covered, best_no, best_trace)
            
        
        
        return best_sol, (best_covered, best_no, best_trace)

class GeneticAlgorithms(object):
    def __init__(self, objective_funtion, pop_size=50, BW=0.2, p_c=0.8, p_m=0.025):
        self._obj_func = objective_funtion
        self.pop_size = pop_size
        self.BW = BW
        self.p_c = p_c
        self.p_m = p_m
        #self.size = random.randint(min(min_noS), max(max_noS))
        self.size = 15

        self.population = list() # []
        self.history = list()

    def _random_selection(self):
        if random.random() < 0.8:
            choice = random.randint(0, 1) # type of sensor 
            if choice == 0:
                x = lower[0][0] + (upper[0][0] - lower[0][0])*random.random()
                y = lower[0][1] + (upper[0][1] - lower[0][1])*random.random()
            else:
                x = lower[1][0] + (upper[1][0] - lower[1][0])*random.random()
                y = lower[1][1] + (upper[1][1] - lower[1][1])*random.random()
        else:
            x = -1
            y = -1
        return [x, y]

    def _initialize(self, pop_size, size):
        for _ in range(pop_size):
            chromosomes = list()
            for _ in range(size):
                chromosomes.append(self._random_selection())
            
            fitness = self._obj_func.get_fitness(chromosomes)

            self.population.append((chromosomes, fitness)) # chromosomes is sensor with cordination [x, y]
            # fitness = best_sol, (best_covered, best_no, best_trace)

        self.history.append({'gen': 0, 'chromosomes': self.population})

    def _get_best(self, population):
        fitness = population[0]
        for gen in population:
            if gen[1][0] > fitness[1][0]:
                fitness = gen

        return fitness


    def _selection(self, new_population, tourn_size):
        list_tourn = random.sample(new_population, tourn_size)
        selector = self._get_best(list_tourn)
        self.population.remove(selector)
        return selector        

    def _crossover(self, parent1, parent2):
        sensor_parent1 = parent1[0]
        sensor_parent2 = parent2[0]

        num = len(sensor_parent1)
        rand_pos = random.randint(0, num-1)
        for i in range(rand_pos, num):
            sensor_parent1[i] = sensor_parent2[i]
            sensor_parent2[i] = sensor_parent1[i]

        fitness1 = self._obj_func.get_fitness(sensor_parent1)
        fitness2 = self._obj_func.get_fitness(sensor_parent2)
        #parent1[0] = sensor_parent1
        child1 = (sensor_parent1, fitness1)
        child2 = (sensor_parent2, fitness2)
        return child1, child2

    def _mutation(self, gen, rate):
        for i in range(len(gen[0])):
            if random.random() < rate:
                gen[0][i][0] = gen[0][i][0] + self.BW * random.random()
                gen[0][i][1] = gen[0][i][1] + self.BW * random.random()
        return gen


    def run(self, steps=100):
        self._initialize(self.pop_size, self.size)
        generation = 0
        for i in range(steps):
            # NEW_POPULATION
            tmp_population = list()
            pop_size = len(self.population)
            elitism_num = pop_size // 2

            new_population = list()
            #Elitism
            # for _ in range(elitism_num):
            #     best_gen = self._get_best(self.population)
            #     tmp_population.append(best_gen)
            #     self.population.remove(best_gen)

            # Crossover
            for _ in range(elitism_num, pop_size):
                parent1 = self._selection(self.population, len(self.population))
                parent2 = self._selection(self.population, len(self.population))

                if random.random() < self.p_c:
                    child1, child2 = self._crossover(parent1, parent2)
                    new_population.append(child1)
                    new_population.append(child2)
                else:
                    new_population.append(parent1)
                    new_population.append(parent2)
                    # self.population.remove(parent1)
                    # self.population.remove(parent2)

            # Mutation 
            for i in range(0, pop_size):
               new_population[i] = self._mutation(new_population[i], self.p_m)
            
            self.population = new_population
            generation+=1
            
            best_gen = self._get_best(self.population)
            print("gen", generation, " with best =", best_gen[1][0], "cover: ", best_gen[1][1][0],
             " used: ", best_gen[1][1][1], "type: ", best_gen[1][1][2])
        

obj = ObjectiveFunction(targets=targets)
ga = GeneticAlgorithms(obj)
print("______________________START____________________________")
ga.run(10)
print("_______________________END_____________________________")

