"""
Configuración de toolbox de DEAP.
Configuración de inicialización, función fitness y operadores de reproducción y mutación.
"""

import random
import time
import numpy as np

from typing import Callable, Tuple, Any

from deap import base, creator, tools

import torch
from torch.utils.data import DataLoader
from torch import nn

from .types import Capa, Gen, NeuralNetwork

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))


"""
Creator tiene definiciones a nivel global
"""

# Define la multifunción fitness que evalúa cuatro cosas:
#   1º parámetro: minimiza el error
#   2º parámetro: minimiza el número de neuronas
#   3º parámetro: minimiza el número de capas
#   4º parámetro: minimiza el valor medio de pérdida
creator.create(
    "FitnessMulti",
    base.Fitness,
    weights=(-1.0, -0.5, -0.5, -0.9)
)

# Define cuál es el gen y cuál es la función que trabaja sobre él
creator.create(
    "Individual",
    Gen,
    fitness=creator.FitnessMulti
)

''' Empiezan las funciones auxiliares '''

def train(
    dataloader: DataLoader,
    model: NeuralNetwork,
    loss_fn: nn.CrossEntropyLoss,
    optimizer: torch.optim.SGD,
    device: str = "cpu"
) -> None:
    ''' Función de entrenamiento de la red neuronal '''
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        """ if batch % 100 == 0:
            loss = loss.item()
            print(f"Train info. \n Loss: {loss:>7f}") """


def test(
    dataloader: DataLoader,
    model: NeuralNetwork,
    loss_fn: nn.CrossEntropyLoss,
    device: str = "cpu"
) -> Tuple[float, float]:

    ''' Función de evaluación del test '''

    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    """ print(
        f"Test Error. \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n") """
    return (100*correct, test_loss)


def inicializador_gen(
    clase_gen: Callable,
    capas_entrada_salida: Tuple[int, int],
    rango_num_neuronas: Tuple[int, int],
    rango_num_capas: Tuple[int, int]
) -> Any:
    '''
    Inicializa los individuos de manera uniforme (capas y número de neuronas)
    '''

    #  Inicializa una lista de capas ocultas de manera uniforme
    hidden_layers = np.random.randint(
        rango_num_neuronas[0],
        rango_num_neuronas[1] + 1,
        random.randint(*rango_num_capas),
    ).tolist()

    # Crea el individuo de la clase Gen, enviándole como parámetro la capa de entrada, salida y las ocultas recién inicializadas
    return clase_gen(capas_entrada_salida[0], capas_entrada_salida[1], hidden_layers)


def evaluador_gen(
    gen: Gen,  # Gen individual a evaluar
    trn: DataLoader,  # Datos de Train
    tst: DataLoader,  # Datos de Test
    max_epochs: int # Número de epochs a realizar
) -> Tuple[float, int, int, float]:
    '''
    Evaluación de cada gen individual
    '''

    # Medición del tiempo
    tiempo_inicial = time.perf_counter()

    # Crea una lista con el número de neuronas de cada capa
    tamanos = [capa.neuronas_salida for capa in gen.capas]

    # Crea un modelo con la información del gen individual
    modelo = NeuralNetwork(gen).to(device)

    # Se define la función de pérdida y backpropagation para optimizar a través del cálculo de las derivadas
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(modelo.parameters(), lr=1e-3)

    epochs = max_epochs  # número de veces que pasarán los datos por la RN
    acc, avg_loss = 0.0, 0.0  # valores iniciales de las métricas
    for t in range(epochs):
        #print(f"Epoch {t+1}\n-------------------------------")
        train(trn, modelo, loss_fn, optimizer, device)  # entrena
        acc, avg_loss = test(tst, modelo, loss_fn, device)  # testea

    # Para optimizar la RN, se calculan los tamaños: cantidad de capas y total de neuronas
    num_neuronas = sum(tamanos)
    num_capas = len(tamanos)

    gen.modelo = modelo

    # Calcula el error porcentual que presenta la RN
    error_perc = 100.00 - acc

    """print(
        f"{'error = ':>22}{error_perc:.2f} %\n",
        f"{'nº neuronas ocultas = ':>22}{num_neuronas:.2f}\n",
        f"{'nº capas ocultas = ':>22}{num_capas:.2f}\n",
        f"{'avg_loss = ':>22}{avg_loss:.5f}\n",
        f"{'evaluation time = ':>22}{time.perf_counter() - tiempo_inicial:.2f} sec"
    ) """

    # Devuelve las métricas del entrenamiento actual
    return (error_perc, num_neuronas, num_capas, avg_loss)


def operador_crossover(
    ind1: Gen,
    ind2: Gen
) -> Tuple[int, int]:
    # Choose randomly the layer index to swap. If the hidden layers of any of
    # the two individuals are constant, swap neurons from the output layer
    # neuron in the output layer.
    layer_index = random.randint(0, len(ind1) - 1)

    cx_pts = random.sample(range(len(ind1.capas[layer_index].sesgos)), 2)

    (
        ind1.capas[layer_index].pesos[:, cx_pts[0]: cx_pts[1]],
        ind2.capas[layer_index].pesos[:, cx_pts[0]: cx_pts[1]],
    ) = (
        ind2.capas[layer_index].pesos[:, cx_pts[0]: cx_pts[1]].copy(),
        ind1.capas[layer_index].pesos[:, cx_pts[0]: cx_pts[1]].copy(),
    )
    (
        ind1.capas[layer_index].sesgos[cx_pts[0]: cx_pts[1]],
        ind2.capas[layer_index].sesgos[cx_pts[0]: cx_pts[1]],
    ) = (
        ind2.capas[layer_index].sesgos[cx_pts[0]: cx_pts[1]].copy(),
        ind1.capas[layer_index].sesgos[cx_pts[0]: cx_pts[1]].copy(),
    )

    return cx_pts, layer_index


def mutador_capa(
    individual: Gen,
    rango_neuronas: Tuple[int, int]
) -> int:
    # Elegir aleatoriamente si agregar o quitar una capa, habiendo al menos dos en el modelo
    choice = 1 if len(individual) <= 2 else random.choice((-1, 1))

    difference = 0

    if choice > 0:
        # Elegir un número aleatorio de neuronas
        new_layer_output_neurons = random.randint(*rango_neuronas)
        # Obtener el número de neuronas de la última capa oculta
        previous_layer_output = individual.capas[-2].neuronas_salida
        # Agregar una última capa en el modelo
        individual.agregar_capa(
            Capa(
                nombre=f"Capa{len(individual)}",
                neuronas_entrada=previous_layer_output,
                neuronas_salida=new_layer_output_neurons,
            )
        )

        # Obtener las diferencias entre la nueva capa y la de salida para aplicar los cambios necesarios para los cálculos
        output_layer_input_neurons = individual.capas[-1].pesos.shape[0]
        difference = new_layer_output_neurons - output_layer_input_neurons

        # Agregar los valores de las neuronas a la capa elegida
        if difference > 0:
            next_layer_neurons = len(individual.capas[-1].sesgos)
            individual.capas[-1].pesos = np.append(
                individual.capas[-1].pesos,
                np.random.uniform(-1.0, 1.0, (difference, next_layer_neurons)),
                axis=0,
            )
        # Eliminar los valores de las neuronas a la capa elegida
        elif difference < 0:
            individual.capas[-1].pesos = np.delete(
                individual.capas[-1].pesos,
                slice(
                    output_layer_input_neurons + difference,
                    output_layer_input_neurons,
                ),
                axis=0,
            )
    else:
        # Obtener los valores siguientes y eliminar la capa
        removed_predecessor_units = individual.capas[-3].neuronas_salida
        del individual.capas[-2]

        # Calcular la diferencia entre la capa a eliminary la de salida
        output_layer_input_len = individual.capas[-1].pesos.shape[0]
        difference = removed_predecessor_units - output_layer_input_len

        # Agregar las nauronas de entrada necesarias
        if difference > 0:
            next_layer_neurons = len(individual.capas[-1].sesgos)
            individual.capas[-1].pesos = np.append(
                individual.capas[-1].pesos,
                np.random.uniform(-0.5, 0.5, (difference, next_layer_neurons)),
                axis=0,
            )
        # Eliminar el excedente
        elif difference < 0:
            individual.capas[-1].pesos = np.delete(
                individual.capas[-1].pesos,
                slice(
                    output_layer_input_len + difference, output_layer_input_len
                ),
                axis=0,
            )

    # Actualizar las neuronas de entrada de la capa de salida
    individual.capas[-1].neuronas_entrada += difference

    return choice


def mutador_neurona(
    individual: Gen
) -> int:
    # Las capas de entrada y salida son fijas, por lo que se agregan o quitan son las capas ocultas
    layer_index = random.randint(0, len(individual) - 2)

    # Elige una capa al azar para añadir una neurona más
    choice = 1 if len(
        individual.capas[layer_index].sesgos) <= 2 else random.choice((-1, 1))

    if choice > 0:
        # Toma la información de la capa anterior para crear el nuevo individuo
        previous_layer_neurons = individual.capas[layer_index].pesos.shape[0]

        # Agregar la neuva neurona a los pesos y sesogs de la capa seleccionada
        individual.capas[layer_index].pesos = np.append(
            individual.capas[layer_index].pesos,
            np.random.uniform(-0.5, 0.5, (previous_layer_neurons, 1)),
            axis=1,
        )
        individual.capas[layer_index].sesgos = np.append(
            individual.capas[layer_index].sesgos,
            [random.uniform(-0.5, 0.5)],
            axis=0,
        )
        # Añadir la nueva neurona a la capa escogida
        next_layer_neurons = len(individual.capas[layer_index + 1].sesgos)
        individual.capas[layer_index + 1].pesos = np.append(
            individual.capas[layer_index + 1].pesos,
            np.random.uniform(-0.5, 0.5, (1, next_layer_neurons)),
            axis=0,
        )
    else:
        # Eliminar la última neurona de los pesos y los sesgos
        individual.capas[layer_index].pesos = np.delete(
            individual.capas[layer_index].pesos, -1, axis=1
        )
        individual.capas[layer_index].sesgos = np.delete(
            individual.capas[layer_index].sesgos, -1, axis=0
        )
        # Eliminar la última neurona de la próxima capa
        individual.capas[layer_index + 1].pesos = np.delete(
            individual.capas[layer_index + 1].pesos, -1, axis=0
        )

    # Actualizar el valor de la capa actual y la próxima
    individual.capas[layer_index].neuronas_salida += choice
    individual.capas[layer_index + 1].neuronas_entrada += choice

    return choice


def mutador_pesos_sesgos(
    individual: Gen,
    attribute: str,  # define qué es lo que se va a mutar
    gen_prob: float
) -> int:
    ''' Operador de mutación de pesos y sesgos '''
    mutated_genes = 0

    for capa in individual.capas:
        weights = getattr(capa, attribute)
        weights_shape = weights.shape
        mask = np.random.rand(*weights_shape) < gen_prob
        mutated_genes += np.count_nonzero(mask)
        mutations = np.random.uniform(-0.5, 0.5, weights_shape)
        mutations[~mask] = 0
        weights += mutations

    return mutated_genes


def configure_toolbox(
    dataset: Tuple[DataLoader, DataLoader],
    nin_nout: Tuple[int, int],
    rango_neuronas: Tuple[int, int],
    rango_capas: Tuple[int, int],
    prob_mut_sesgos: float,
    prob_mut_pesos: float,
    max_epochs_ml: int
):
    '''
    Función de inicialización del algoritmo genético
    '''
    toolbox = base.Toolbox()

    # Registra el individuo sobre el que trabajar
    toolbox.register(
        "individuo",
        inicializador_gen,
        creator.Individual,
        nin_nout,
        rango_neuronas,
        rango_capas
    )

    # Define la población inicial del genético
    toolbox.register(
        "poblacion",
        tools.initRepeat,
        list,
        toolbox.individuo
    )

    # Define cuál es la función de evaluación y sobre qué datasets se aplica
    toolbox.register(
        "evaluar",
        evaluador_gen,
        trn=dataset[0],
        tst=dataset[1],
        max_epochs=max_epochs_ml
    )

    # Define el operador de reproducción: crossover
    toolbox.register(
        "cruzar",
        operador_crossover
    )

    # Define la función de mutación de sesgos
    toolbox.register(
        "mutar_sesgos",
        mutador_pesos_sesgos,
        attribute="sesgos",
        gen_prob=prob_mut_sesgos
    )

    # Define la función de mutación de pesos
    toolbox.register(
        "mutar_pesos",
        mutador_pesos_sesgos,
        attribute="pesos",
        gen_prob=prob_mut_pesos,
    )

    # Define la función de mutación de neuronas
    toolbox.register(
        "mutar_neurona",
        mutador_neurona
    )

    # Define la función de mutación de capas
    toolbox.register(
        "mutar_capa",
        mutador_capa,
        rango_neuronas=rango_neuronas
    )

    return toolbox
