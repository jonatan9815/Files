import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from mlxtend.feature_selection import SequentialFeatureSelector as SFS

# Configurar estilo de gráficos
sns.set(style="whitegrid")

# Cargar los datos
file_path = r"C:\Users\jonat\Downloads\visual\A1.4 Vino Tinto.csv"
df = pd.read_csv(file_path)

# Mostrar dimensiones y primeras filas del DataFrame
print("### **Reporte de Resultados: Actividad A1.4 - Selección de Características**")
print("\n#### **1. Importación de los Datos**")
print(f"- **Dimensiones del DataFrame**: {df.shape}")
print("- **Primeras 5 filas de datos:**")
print(df.head())

# Separar las variables predictoras (X) y la variable objetivo (y)
X = df.drop(columns=['calidad'])  # Variable objetivo: calidad
y = df['calidad']

# División de los datos en entrenamiento (80%) y prueba (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\n#### **2. Separación de Datos en Entrenamiento y Prueba**")
print(f"- **Entrenamiento (X_train)**: {X_train.shape}")
print(f"- **Prueba (X_test)**: {X_test.shape}")

# Histograma de calidad del vino
plt.figure(figsize=(8, 5))
sns.histplot(y, bins=10, kde=True, color="blue")
plt.title("Distribución de Calidad del Vino")
plt.xlabel("Calidad")
plt.ylabel("Frecuencia")
plt.savefig("histograma_calidad.png")
plt.show()

# Configuración del modelo de regresión
model = LinearRegression()

# ======= SELECCIÓN HACIA ADELANTE =======
sfs_forward = SFS(model, k_features=(2, 8), forward=True, floating=False, scoring='r2', cv=10)
sfs_forward.fit(X_train, y_train)
selected_features_forward = list(sfs_forward.k_feature_names_)

print("\n#### **3. Selección Hacia Adelante**")
print(f"- **Características seleccionadas**: {selected_features_forward}")

# Entrenar el modelo con selección hacia adelante
X_train_forward = X_train[selected_features_forward]
X_test_forward = X_test[selected_features_forward]
model.fit(X_train_forward, y_train)
y_pred_forward = model.predict(X_test_forward)

# Evaluar el modelo con R² y MSE
r2_forward = r2_score(y_test, y_pred_forward)
mse_forward = mean_squared_error(y_test, y_pred_forward)
print("\n#### **4. Modelo con Selección Hacia Adelante**")
print(f"- **R cuadrada del modelo**: {r2_forward:.4f}")
print(f"- **Error Cuadrático Medio (MSE)**: {mse_forward:.4f}")

# ======= SELECCIÓN HACIA ATRÁS =======
sfs_backward = SFS(model, k_features=(2, 5), forward=False, floating=False, scoring='r2', cv=10)
sfs_backward.fit(X_train[selected_features_forward], y_train)
selected_features_backward = list(sfs_backward.k_feature_names_)

print("\n#### **5. Selección Hacia Atrás**")
print(f"- **Características seleccionadas**: {selected_features_backward}")

# Entrenar el modelo con selección hacia atrás
X_train_backward = X_train[selected_features_backward]
X_test_backward = X_test[selected_features_backward]
model.fit(X_train_backward, y_train)
y_pred_backward = model.predict(X_test_backward)

# Evaluar el modelo con R² y MSE
r2_backward = r2_score(y_test, y_pred_backward)
mse_backward = mean_squared_error(y_test, y_pred_backward)
print("\n#### **6. Modelo con Selección Hacia Atrás**")
print(f"- **R cuadrada del modelo**: {r2_backward:.4f}")
print(f"- **Error Cuadrático Medio (MSE)**: {mse_backward:.4f}")

# ======= COMPARACIÓN DE MODELOS =======
r2_difference = r2_forward - r2_backward
r2_difference_pct = (r2_difference / r2_forward) * 100  # Diferencia en porcentaje

print("\n### **Comparación de Modelos**")
print(f"- **Modelo con selección hacia adelante**: R² = {r2_forward:.4f}, MSE = {mse_forward:.4f}")
print(f"- **Modelo con selección hacia atrás**: R² = {r2_backward:.4f}, MSE = {mse_backward:.4f}")
print(f"- **Diferencia entre los modelos (R²)**: {r2_difference:.4f}")
print(f"- **Diferencia en porcentaje (R²)**: {r2_difference_pct:.2f}%")

# Gráfico de dispersión de predicciones vs valores reales
plt.figure(figsize=(8, 5))
sns.scatterplot(x=y_test, y=y_pred_forward, label="Selección Adelante", color="blue", alpha=0.6)
sns.scatterplot(x=y_test, y=y_pred_backward, label="Selección Atrás", color="red", alpha=0.6)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='black', linestyle='--')  # Línea perfecta
plt.xlabel("Calidad Real")
plt.ylabel("Predicción del Modelo")
plt.title("Comparación de Predicciones vs. Valores Reales")
plt.legend()
plt.savefig("comparacion_modelos.png")
plt.show()

# ======= CONCLUSIÓN =======
print("\n### **Conclusión sobre los Modelos**")
if r2_forward > r2_backward:
    print("- ✅ El modelo con selección hacia adelante es superior en precisión.")
else:
    print("- 🔵 El modelo con selección hacia atrás es mejor en precisión.")

print("\n### **Opinión Final**")
print("- Ambos modelos tienen un R² relativamente bajo (<0.5), lo que indica que la calidad del vino puede depender de más factores no considerados.")
print("- Se podrían probar modelos más avanzados como árboles de decisión, random forests o redes neuronales.")
print("- También se puede mejorar el rendimiento con técnicas de regularización como Ridge o Lasso.")