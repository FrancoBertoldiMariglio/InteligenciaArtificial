import numpy as np
from tensorflow.keras.layers import Dense
from tensorflow.keras import Sequential

# Datos de entrada (XOR)
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])

# Salidas esperadas
y = np.array([[0], [1], [1], [0]])

# Crear el modelo secuencial
model = Sequential()

# Añadir capa oculta con 3 neuronas y activación 'sigmoid'
model.add(Dense(3, input_dim=2, activation='sigmoid'))

# Añadir capa de salida con 1 neurona y activación 'sigmoid'
model.add(Dense(1, activation='sigmoid'))

# Compilar el modelo con una función de pérdida de entropía binaria y el optimizador 'adam'
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Entrenar el modelo, con resultados por cada epoch
model.fit(X, y, epochs=10000, verbose=1)

# Probar el modelo con las entradas XOR
predictions = model.predict(X)

# Mostrar las predicciones finales
print("\nPredicciones finales:")
for i in range(len(X)):
    print(f"Entrada: {X[i]} -> Predicción: {predictions[i][0]:.4f} -> Salida esperada: {y[i][0]}")
