"""Util functions used in the GA."""

from typing import Tuple

import random

from typing import Callable, Tuple

import numpy as np

from deap import base

from torch.utils.data import DataLoader

from .types import Gen
from .toolbox import evaluador_gen

def evaluate_population(population: list, evaluate_fn: Callable) -> None:
    """Evaluate the population.
    Apply the evaluation method to every individual of the population.
    :param population: list of individuals.
    :param evaluate_fn: function to evaluate the population.
    """
    fitnesses = map(evaluate_fn, population)

    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit


def apply_crossover(
    population: list, cx_prob: float, crossover_fn: Callable
) -> int:
    crossed_individuals = 0

    for crossover_index, (child1, child2) in enumerate(
        zip(population[::2], population[1::2])
    ):
        if random.random() < cx_prob and child1.can_mate(child2):
            crossed_individuals += 1
            cx_pt, layer_index = crossover_fn(child1, child2)
            del child1.fitness.values
            del child2.fitness.values
            #print(
            #    f"    Applying crossover for the {crossover_index} couple "
            #    f"({crossover_index * 2}, {crossover_index * 2+1})).\n"
            #    f"        Crossed from neuron {cx_pt[0]} to {cx_pt[1]} in "
            #    f"layer {layer_index}"
            #)

    return crossed_individuals


def apply_mutation(
    population: list,
    toolbox: base.Toolbox,
    mut_neurons_prob: float,
    mut_layers_prob: float,
) -> int:
    mutated_individuals = 0

    for index, mutant in enumerate(population):
        mut_bias_genes = mut_weights_genes = neuron_diff = layer_diff = 0

        mut_bias_genes = toolbox.mutar_sesgos(mutant)
        mut_weights_genes = toolbox.mutar_pesos(mutant)
        mutated_individuals += 1
        del mutant.fitness.values

        # Ensure that we don't modify the hidden layers if they are constant
        if random.random() < mut_neurons_prob:
            neuron_diff = toolbox.mutar_neurona(mutant)

        if random.random() < mut_layers_prob:
            layer_diff = toolbox.mutar_capa(mutant)

        #print(
        #    f"    For individual {index}:\n"
        #    f"        {mut_bias_genes} mutated bias genes\n"
        #    f"        {mut_weights_genes} mutated weights genes\n"
        #    f"        {neuron_diff} neuron changes\n"
        #    f"        {layer_diff} layer changes\n"
        #)

    return mutated_individuals


def finished_generation_summary(
    current_generation: int, population: list, best_fit: tuple
):
    fits = np.array([ind.fitness.values for ind in population])
    table = [
        ["Statistic", "Accuracy error %", "Neuron/Layer score", "F2 score"],
        ["Max", *fits.max(0)],
        ["Avg", *fits.mean(0)],
        ["Min", *fits.min(0)],
        ["Std", *fits.std(0)],
        ["Best", *best_fit],
    ]
    #print(f"    Summary of generation {current_generation}:")
    #print(table)


def finished_algorithm_summary(
    initial_population, final_population, best_individual_selector
):
    initial_pop_table = []
    final_pop_table = []
    final_pop_neurons_table = []
    final_pop_ind_layer_list = [
        [capa.neuronas_salida for capa in ind.capas[:-1]]
        for ind in final_population
    ]
    max_layer_ind = max(map(len, final_pop_ind_layer_list))
    final_pop_layer_list = np.array(
        [
            layer_list + [0] * (max_layer_ind - len(layer_list))
            for layer_list in final_pop_ind_layer_list
        ]
    )

    for index, individual in enumerate(
        best_individual_selector(initial_population, len(initial_population))
    ):
        initial_pop_table.append(
            [
                str(index),
                str(individual.fitness.values[0]),
                str(individual.fitness.values[1]),
            ]
        )

    print("Initial population fitness values:")
    print(initial_pop_table)

    for index, individual in enumerate(
        best_individual_selector(final_population, len(final_population))
    ):
        final_pop_table.append(
            [
                str(index),
                str(individual.fitness.values[0]),
                str(individual.fitness.values[1]),
            ]
        )

    print("Final population fitness values:")
    print(final_pop_table)

    final_pop_neurons_table.append(
        [
            "Statistic",
            *[f"Hidden layer {idx}" for idx in range(max_layer_ind)],
        ]
    )
    final_pop_neurons_table.append(["Max", *final_pop_layer_list.max(0)])
    final_pop_neurons_table.append(["Mean", *final_pop_layer_list.mean(0)])
    final_pop_neurons_table.append(["Min", *final_pop_layer_list.min(0)])
    final_pop_neurons_table.append(["Std", *final_pop_layer_list.std(0)])
    print("Final population layer neurons statistics:")
    print(final_pop_neurons_table)


def test_individual(
    individual: Gen,
    dataset: Tuple[DataLoader, DataLoader],
    max_epochs: int
) -> Tuple[float, float]:
    tst_scores = evaluador_gen(
        individual, dataset[0], dataset[1], max_epochs
    )
    return tst_scores