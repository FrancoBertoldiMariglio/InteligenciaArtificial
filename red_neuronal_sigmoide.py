import numpy as np


# Función de activación sigmoide y su derivada
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


# Datos de entrada (XOR)
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])

# Salidas esperadas
y = np.array([[0], [1], [1], [0]])

# Semilla aleatoria para reproducibilidad
np.random.seed(42)

# Inicialización de los pesos de forma aleatoria
weights_input_hidden = np.random.rand(2, 3)  # 2 entradas -> 3 neuronas ocultas
weights_hidden_output = np.random.rand(3, 1)  # 3 ocultas -> 1 salida

# Tasa de aprendizaje
learning_rate = 0.1

# Número de iteraciones de entrenamiento
epochs = 100000

# Entrenamiento de la red neuronal
for epoch in range(epochs):
    # Propagación hacia adelante (forward propagation)
    hidden_input = np.dot(X, weights_input_hidden)  # Entrada a la capa oculta
    hidden_output = sigmoid(hidden_input)  # Salida de la capa oculta

    final_input = np.dot(hidden_output, weights_hidden_output)  # Entrada a la capa de salida
    final_output = sigmoid(final_input)  # Salida de la red neuronal

    # Error en la salida
    error = y - final_output

    if epoch % 1000 == 0:
        print(f"Iteración {epoch}, Error: {np.mean(np.abs(error))}")

    # Retropropagación (backpropagation)
    # Cálculo del gradiente para la capa de salida
    d_output = error * sigmoid_derivative(final_output)

    # Cálculo del gradiente para la capa oculta
    error_hidden_layer = d_output.dot(weights_hidden_output.T)
    d_hidden = error_hidden_layer * sigmoid_derivative(hidden_output)

    # Actualización de los pesos
    weights_hidden_output += hidden_output.T.dot(d_output) * learning_rate
    weights_input_hidden += X.T.dot(d_hidden) * learning_rate

# Resultados finales después del entrenamiento
print("\nResultados finales:")
for i in range(len(X)):
    hidden_output = sigmoid(np.dot(X[i], weights_input_hidden))
    final_output = sigmoid(np.dot(hidden_output, weights_hidden_output))
    print(f"Entrada: {X[i]} -> Predicción: {final_output[0]:.4f} -> Salida esperada: {y[i][0]}")
