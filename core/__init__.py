"""Multilayer perceptron optimization via genetic algorithms."""

import random
import time

from typing import Tuple

import numpy as np

from deap import tools

from torch.utils.data.dataloader import DataLoader

from .toolbox import configure_toolbox
from .types import Gen, NeuralNetwork
from .utils import (
    apply_crossover,
    apply_mutation,
    evaluate_population,
    finished_algorithm_summary,
    finished_generation_summary,
    test_individual,
)

__all__ = ["genetic_algorithm", "Gen", "test_individual", "NeuralNetwork"]

def genetic_algorithm(
    dataset: Tuple[DataLoader, DataLoader],

    nin_nout: Tuple[int, int],
    rango_neuronas: Tuple[int, int],
    rango_capas: Tuple[int, int],
    prob_mut_sesgos: float,
    prob_mut_pesos: float,
    max_epochs_nn: int,

    prob_cruce: float,
    prob_mut_neurona: float,
    prob_mut_capa: float,

    tamano_poblacion: int,
    max_generaciones: int,
    seed: int = None,
) -> Tuple[Gen, Gen]:
    np.set_printoptions(precision=5, floatmode="fixed")
    if seed:
        np.random.seed(seed)
        random.seed(seed)

    # Configure dataset related variables
    toolbox = configure_toolbox(
        dataset,
        nin_nout,
        rango_neuronas,
        rango_capas,
        prob_mut_sesgos,
        prob_mut_pesos,
        max_epochs_nn
    )

    # --------------------------------
    # Algorithm start
    # --------------------------------
    time_start = time.perf_counter()
    population = toolbox.poblacion(n=tamano_poblacion)

    evaluate_population(population, toolbox.evaluar)

    initial_population = population[:]

    current_generation = 0
    current_gen_best_fit = max([ind.fitness for ind in population])
    previous_best_fit = None

    try:
        # Begin the evolution
        while (
            current_gen_best_fit.values != (0.0, 1.0, 1.0, 0.0)
            and current_generation < max_generaciones
        ):

            # Check if no score improvement has been made
            if current_generation % 10 == 0:
                if (
                    previous_best_fit
                    and current_gen_best_fit <= previous_best_fit
                ):
                    print(
                        "The fitness has not improved in 10 generations:\n"
                        f"\tPrevious best fit: {previous_best_fit}\n"
                        f"\tCurrent best fit: {current_gen_best_fit}\n"
                        "Exiting..."
                    )
                    break

                previous_best_fit = current_gen_best_fit

            # A new generation
            current_generation = current_generation + 1

            # Select the best individuals for the offspring
            best_population_individuals = tools.selBest(
                population, int(len(population) / 2)
            )
            # Clone the selected individuals
            offspring = list(map(toolbox.clone, best_population_individuals))

            # --------------------------------
            # Operators
            # --------------------------------

            crossed_individuals = apply_crossover(
                offspring, prob_cruce, toolbox.cruzar
            )

            mutated_individuals = apply_mutation(
                offspring, toolbox, prob_mut_neurona, prob_mut_capa
            )

            # --------------------------------
            # Evaluation
            # --------------------------------

            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]

            evaluate_population(invalid_ind, toolbox.evaluar)

            # Replace the worst individuals from the previous population with
            # the mutated ones. In other words, create the new offspring
            # from the previous population best individual plus the mutated
            # ones
            population = best_population_individuals + offspring
            current_gen_best_fit = tools.selBest(population, 1)[0].fitness

            finished_generation_summary(
                current_generation, population, current_gen_best_fit.values
            )
    except KeyboardInterrupt:
        print("Stopping the algorithm...")

    elapsed_time = time.perf_counter() - time_start
    print(
        f"-- Finished evolution with the generation {current_generation} in "
        f"{elapsed_time:.2f} seconds."
    )
    finished_algorithm_summary(initial_population, population, tools.selBest)
    best_final_individual = tools.selBest(population, 1)[0]
    tst_scores = test_individual(best_final_individual, dataset, max_epochs_nn)

    return best_final_individual.modelo, tst_scores