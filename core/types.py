import torch
from torch import nn

from typing import List

import numpy as np

class Capa:

    ''' Clase que representa una Capa '''

    def __init__(
            self,
            nombre: str,
            neuronas_entrada: int,
            neuronas_salida: int,
            activacion: str = "relu" # función de activación por defecto
    ) -> None:
        '''
            Esta clase permite definir en una abstracción a cada una de las capas de una RN
            Para instanciar una capa es necesario definirle los siguientes atributos: 
                nombre (str): [description]
                neuronas_entrada (int): [description]
                neuronas_salida (int): [description]
                activacion (str, optional): [description]. Defaults to "relu"
        '''
        # Inicializaciones de los parámetros de clase
        self.nombre = nombre
        self.neuronas_entrada = neuronas_entrada
        self.neuronas_salida = neuronas_salida
        self.activacion = activacion  # si es null, se mantiene relu

        # Toma los valores de entrada y salida e inicializa los pesos y el sesgo
        self.pesos = np.random.uniform(-1.0, 1.0,
                                       size=(neuronas_entrada, neuronas_salida))
        self.sesgos = np.random.uniform(-1.0, 1.0, size=(neuronas_salida, ))

    def __str__(self) -> str:
        """Serialize as a string the object."""
        str_representation = (
            f"Layer {self.nombre}:"
            f"\n-- Pesos --\n{str(self.pesos)}"
            f"\n-- Sesgos --\n{str(self.sesgos)}"
        )

        return str_representation

    def __repr__(self) -> str:
        """
        Convertidor de objetos a string
        """
        return str(self)

class Gen:

    ''' Clase que representa un Gen '''

    def __init__(
        self,  # indica que es un método
        neuronas_entrada:int,  # neuronas de entrada
        neuronas_salida:int,  # neuronas de salida
        capas_ocultas:List[int],  # capas con la cantidad de neuronas
    ) -> None:
        """
        Constructora del Gen: instancia un objeto de la clase Gen que tendrá la definición de una RNA

        Args:
            neuronas_entrada (int): número de neuronas en la capa de entrada
            neuronas_salida (int): número de neuronas en la capa de salida
            capas_ocultas (List[int]): lista con el número de neuronas en cada capa, cada elemento representa además una capa oculta
        """
        self.modelo = None
        self.capas: List[Capa] = []
        proxima_capa = neuronas_entrada

        for i, numero_neuronas in enumerate(capas_ocultas):
            self.capas.append(
                Capa(
                    nombre=f"Capa{i}",
                    neuronas_entrada=proxima_capa,
                    neuronas_salida=numero_neuronas
                )
            )

            # la capa actual será la capa de entrada de la próxima capa oculta
            proxima_capa = numero_neuronas

        # Caso particular para añadir capa de salida
        self.capas.append(
            Capa(
                nombre="Salida",  # Nombre de la última capa
                # número de neuronas de la capa inmediatamente anterior
                neuronas_entrada=proxima_capa,
                neuronas_salida=neuronas_salida,
                activacion="sigmoid"  # función sigmode
            )
        )

    def __len__(self) -> int:
        """
        Función que calcula la cantidad de capas que tiene un Gen
        """
        return (len(self.capas))

    def agregar_capa(self, capa_nueva: Capa) -> None:
        """
        Función de añadir una capa al Gen (mutación 1)
        """
        self.capas.insert(len(self) - 1, capa_nueva)

    def can_mate(self, other: "Gen") -> bool:
        if len(self) != len(other):
            return False

        for capa1, capa2 in zip(self.capas, other.capas):
            if (
                capa1.pesos.shape != capa2.pesos.shape
                or capa1.sesgos.shape != capa2.sesgos.shape
            ):
                return False

        return True

    def __str__(self) -> str:
        """Serialize as a string the object."""
        str_representation = (
            f"{self.__class__.__name__}:"
        )

        for i, capa in enumerate(self.capas):
            str_representation += (
                f"\nLayer {i}:"
                f"\n-- Pesos --\n{str(capa.pesos)}"
                f"\n-- Sesgos --\n{str(capa.sesgos)}"
            )

        return str_representation

    def __repr__(self) -> str:
        """
        Convertidor de objetos a string
        """
        return str(self)

class NeuralNetwork(nn.Module):
    '''
        Clase que crea modelos de RN, utiliza Module, clase base provista por PyTorch para la construcción de modelos de redes neuronales de una forma más simple 
        Fuente: https://www.programmersought.com/article/36653677206/
    '''

    def __init__(self, gen: Gen):
        '''
            Se construye cada instancia de la clase Neural Network como la secuencia de las operaciones que debe realizar a partir de una entrada, para alcanzar una salida
            Para eso requiere un Gen que defina la estructura de la red neuronal, construyendo así el modelo
        '''
        super(NeuralNetwork, self).__init__()

        # Stack módulos
        partial_stack = []

        # Toma el número de capas y neuronas de cada una
        for capa in gen.capas:
            # Aplicar transformación lineal y agregar al módulo
            partial_stack.append(
                nn.Linear(capa.neuronas_entrada, capa.neuronas_salida))
            # Aplicar función de activación a todas las neuronas de la capa
            partial_stack.append(
                nn.ReLU() if capa.activacion == "relu" else nn.Softmax(dim=1))

        # El iterador que construye el stack de módulos para luego ir a nn.Sequential, almacenando las capas de manera secuencial
        self.stack = nn.Sequential(*partial_stack)

        # Bloque de código turbio, sólo se puede rezar aquí.
        i = 0
        def init_weights(m):
            nonlocal i
            if type(m)==nn.Linear:
                capa = gen.capas[i]
                m.weight = nn.Parameter(torch.Tensor(capa.pesos.T), requires_grad=True)
                m.bias = nn.Parameter(torch.Tensor(capa.sesgos), requires_grad=True)
                i += 1
        self.stack.apply(init_weights)

        del partial_stack

    def forward(self, x):
        '''
            Define el cálculo forward del modelo, es decir, cómo devolver la
            salida del modelo requerida de acuerdo con el cálculo de entrada
        '''
        logits = self.stack(x.float())
        return logits
