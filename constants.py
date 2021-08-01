'''
DEFINICIÓN DE CONSTANTES
'''

# Rango de capas ocultas que puede tener una RN
RANGO_CAPAS = (4, 8) # Min 3 capas ocultas y 1 de salida (por eso 4)

# Rango de neuronas que puede tener cada capa de una RN
RANGO_NEURONAS = (10, 30)

# Probabilidad de mutación de sesgos
PROB_MUT_SESGOS = 0.5

# Probabilidad de mutación de pesos
PROB_MUT_PESOS = 0.5

# Neural Network
MAX_EPOCHS_NN = 100

# Probabilidad de cruce entre genes
PROB_CRUCE = 0.5

# Probabilidad de mutación de una neurona
PROB_MUT_NEURONA = 0.5

# Probabilidad de mutación de capa
PROB_MUT_CAPA = 0.5

# Tamaño de la población inicial
TAMANO_POBLACION = 10

# Máximo de generaciones
MAX_GENERACIONES = 10

# Semilla para fijar los valores generados
SEED = None

# Valor mínimo aceptable
EPSILON = 0.00001

# Porcentaje del dataset para train (1-P_TRAIN para test)
P_TRAIN = 0.80

# Tamaño del lote para entrenamiento
BATCH_SIZE = 10