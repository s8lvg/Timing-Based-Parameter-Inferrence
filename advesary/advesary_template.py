import requests
import json
import random
import pygad
from tqdm import tqdm

def send_request_to_network(session,params):
    response = session.post("http://127.0.0.1:8080/predict", json={"input_values" : params})
    return json.loads(response.text)
    
def random_parameters_request(session):
    params = [random.uniform(0,1) for _ in range(4)]
    return_dict = send_request_to_network(session,params)
    return_dict["input_values"] = params
    return return_dict


function_inputs = [0.0 for _ in range(4)]
desired_output = 44
session = requests.session()
mesurements_repetition = 1000

def fitness_func(solution, solution_idx):
    output = list(solution)
    fitness_list = []
    for i in range(mesurements_repetition):
        fitness = send_request_to_network(session,output)
        fitness_list.append(fitness['prediction_time'])
    return int(sum(fitness_list)/mesurements_repetition)


fitness_function = fitness_func

num_generations = 100
num_parents_mating = 4

sol_per_pop = 8
num_genes = len(function_inputs)

init_range_low = 0
init_range_high = 1

parent_selection_type = "sss"
keep_parents = 1

crossover_type = "single_point"

mutation_type = "random"
mutation_percent_genes = 50

with tqdm(total=num_generations) as pbar:
    ga_instance = pygad.GA(num_generations=num_generations,
                        num_parents_mating=num_parents_mating,
                        fitness_func=fitness_function,
                        sol_per_pop=sol_per_pop,
                        num_genes=num_genes,
                        init_range_low=init_range_low,
                        init_range_high=init_range_high,
                        parent_selection_type=parent_selection_type,
                        keep_parents=keep_parents,
                        crossover_type=crossover_type,
                        mutation_type=mutation_type,
                        mutation_percent_genes=mutation_percent_genes,
                        gene_space={'low':0,'high':1},
                        on_generation=lambda _: pbar.update(1))
    ga_instance.run()

solution, solution_fitness, solution_idx = ga_instance.best_solution()
print("Parameters of the best solution : {solution}".format(solution=solution))
print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
