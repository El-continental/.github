from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Cargar el dataset Iris
data = load_iris()
X = data.data
y = data.target

from sklearn.tree import DecisionTreeClassifier

# División
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Modelo
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Predicción y evaluación
y_pred = model.predict(X_test)
print(f"Precisión (Árbol de Decisión): {accuracy_score(y_test, y_pred)}")

from sklearn.svm import SVC

# División
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Modelo
model = SVC(kernel='linear')
model.fit(X_train, y_train)

# Evaluación
y_pred = model.predict(X_test)
print(f"Precisión (SVM): {accuracy_score(y_test, y_pred)}")


from sklearn.neighbors import KNeighborsClassifier

# División
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Modelo
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

# Evaluación
y_pred = model.predict(X_test)
print(f"Precisión (KNN): {accuracy_score(y_test, y_pred)}")

from sklearn.neural_network import MLPClassifier

# División
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Modelo
model = MLPClassifier(hidden_layer_sizes=(10,), max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# Evaluación
y_pred = model.predict(X_test)
print(f"Precisión (Red Neuronal): {accuracy_score(y_test, y_pred)}")


from sklearn.ensemble import RandomForestClassifier

# División
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Modelo
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluación
y_pred = model.predict(X_test)
print(f"Precisión (Random Forest): {accuracy_score(y_test, y_pred)}")


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
