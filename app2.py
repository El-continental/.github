
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_squared_error, r2_score 


data = sns.load_dataset('iris') 
print(data.head()) 


X = data[['sepal_length', 'sepal_width', 'petal_width']] # Variables predictoras 
y = data['petal_length'] # Variable objetivo 

# Dividir el conjunto de datos en entrenamiento y prueba (80% entrenamiento, 20% prueba) 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 
 
print("Tamaño del conjunto de entrenamiento:", X_train.shape) 
print("Tamaño del conjunto de prueba:", X_test.shape) 
 

model = LinearRegression() 
 
# Entrenar el modelo 
model.fit(X_train, y_train) 
 
print(f"Coeficientes: {model.coef_}") 
print(f"Intersección (intercepto): {model.intercept_}") 

y_pred = model.predict(X_test) 
 
predictions_df = pd.DataFrame({'Real': y_test, 'Predicción': y_pred}) 
print(predictions_df.head()) 

mse = mean_squared_error(y_test, y_pred) 
r2 = r2_score(y_test, y_pred) 
print(f"Error cuadrático medio (MSE): {mse}") 
print(f"Coeficiente de determinación R²: {r2}")

import matplotlib.pyplot as plt

# Graficar los valores reales vs. las predicciones
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color='blue', label='Predicciones')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
         color='red', linewidth=2, label="Línea de referencia")
plt.xlabel("Valores reales")
plt.ylabel("Predicciones")
plt.title("Valores reales vs Predicciones")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()



