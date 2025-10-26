"""
🚕 PROYECTO COMPLETO: PREDICCIÓN DE TARIFAS DE TAXI NYC
========================================================
Dataset real: NYC Taxi Trip Duration (Kaggle) gracias Kaggle por la democratizacion del aprendizaje
Objetivo: Predecir el precio del viaje basado en características del viaje
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #Esto es solo para que no me muestre en la consola las notificaciones

import numpy as np
import pandas as pd
import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

print("=" * 70)
print("🚕 ANÁLISIS Y PREDICCIÓN DE TARIFAS DE TAXI NYC")
print("=" * 70)

# ============================================
# 1. CREAR DATASET REALISTA
# ============================================
print("\n📊 PASO 1: Generando dataset realista...")

# Simulamos 10,000 viajes basados en patrones reales de NYC, Pudieran ser los de Montevideo pero los desconozco
np.random.seed(42)
n_samples = 10000

# Features realistas
data = {
    # Distancia en km (0.5 a 50 km)
    'distance_km': np.random.exponential(scale=5, size=n_samples).clip(0.5, 50),
    
    # Duración en minutos (correlacionada con distancia + trafico)
    'duration_min': None,  # La calculamos despues
    
    # Hora del dia (0-23) 24 horas
    'hour': np.random.randint(0, 24, n_samples),
    
    # Dia de la semana (0=Lun, 6=Dom)
    'day_of_week': np.random.randint(0, 7, n_samples),
    
    # Numero de pasajeros (1-6)
    'passengers': np.random.choice([1, 2, 3, 4, 5, 6], n_samples, 
                                   p=[0.7, 0.15, 0.08, 0.04, 0.02, 0.01]),
    
    # Temperatura en grados Celcius - afecta demanda
    'temperature': np.random.normal(15, 10, n_samples).clip(-5, 35),
    
    # Lluvia (0=no, 1=sí)
    'is_raining': np.random.choice([0, 1], n_samples, p=[0.8, 0.2])
}

df = pd.DataFrame(data)

# Calcular duracion (depende de distancia y condiciones)
base_speed = 30  # km/h base
traffic_factor = np.where(
    (df['hour'] >= 7) & (df['hour'] <= 9) | 
    (df['hour'] >= 17) & (df['hour'] <= 19),
    0.5,  # Rush hour: velocidad reducida
    1.0
)
weather_factor = np.where(df['is_raining'] == 1, 0.7, 1.0)

df['duration_min'] = (df['distance_km'] / (base_speed * traffic_factor * weather_factor) * 60)
df['duration_min'] += np.random.normal(0, 5, n_samples)  # Variabilidad
df['duration_min'] = df['duration_min'].clip(3, 120)

# Calcular PRECIO (nuestra variable objetivo)
# Fórmula realista: Base + por km + por minuto + recargos
base_fare = 3.0
per_km = 2.5
per_min = 0.5
rush_hour_surcharge = np.where(
    (df['hour'] >= 7) & (df['hour'] <= 9) | 
    (df['hour'] >= 17) & (df['hour'] <= 19),
    2.0, 0.0
)
rain_surcharge = np.where(df['is_raining'] == 1, 1.5, 0.0)
night_surcharge = np.where((df['hour'] >= 20) | (df['hour'] <= 6), 1.0, 0.0)

df['fare'] = (
    base_fare + 
    (df['distance_km'] * per_km) + 
    (df['duration_min'] * per_min) + 
    rush_hour_surcharge + 
    rain_surcharge + 
    night_surcharge
)

# Agregar ruido realista
df['fare'] += np.random.normal(0, 2, n_samples)
df['fare'] = df['fare'].clip(5, 200)

print(f"✅ Dataset creado: {len(df):,} viajes")
print(f"   Rango de precios: ${df['fare'].min():.2f} - ${df['fare'].max():.2f}")

# ============================================
# 2. ANÁLISIS EXPLORATORIO DE DATOS (EDA)
# ============================================
print("\n📈 PASO 2: Análisis Exploratorio de Datos")
print("-" * 70)

print("\n🔍 Estadísticas descriptivas:")
print(df.describe().round(2))

print("\n📊 Información del dataset:")
print(df.info())

print("\n🎯 Correlación con el precio:")
correlations = df.corr()['fare'].sort_values(ascending=False)
print(correlations)

# Visualizaciones
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Analisis Exploratorio de Datos - Taxis NYC', fontsize=16, fontweight='bold')

# 1. Distribucion de precios
axes[0, 0].hist(df['fare'], bins=50, color='skyblue', edgecolor='black')
axes[0, 0].set_title('Distribucion de Precios')
axes[0, 0].set_xlabel('Precio ($)')
axes[0, 0].set_ylabel('Frecuencia')
axes[0, 0].axvline(df['fare'].mean(), color='red', linestyle='--', label=f'Media: ${df["fare"].mean():.2f}')
axes[0, 0].legend()

# 2. Distancia vs Precio
axes[0, 1].scatter(df['distance_km'], df['fare'], alpha=0.3, s=10)
axes[0, 1].set_title('Distancia vs Precio')
axes[0, 1].set_xlabel('Distancia (km)')
axes[0, 1].set_ylabel('Precio ($)')

# 3. Duracion vs Precio
axes[0, 2].scatter(df['duration_min'], df['fare'], alpha=0.3, s=10, color='green')
axes[0, 2].set_title('Duracion vs Precio')
axes[0, 2].set_xlabel('Duracion (min)')
axes[0, 2].set_ylabel('Precio ($)')

# 4. Precio por hora del dia
hourly_avg = df.groupby('hour')['fare'].mean()
axes[1, 0].plot(hourly_avg.index, hourly_avg.values, marker='o', linewidth=2, color='purple')
axes[1, 0].set_title('Precio Promedio por Hora')
axes[1, 0].set_xlabel('Hora del día')
axes[1, 0].set_ylabel('Precio Promedio ($)')
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].axvspan(7, 9, alpha=0.2, color='red', label='Rush AM')
axes[1, 0].axvspan(17, 19, alpha=0.2, color='red', label='Rush PM')
axes[1, 0].legend()

# 5. Precio por día de la semana
days = ['Lun', 'Mar', 'Mie', 'Jue', 'Vie', 'Sab', 'Dom']
daily_avg = df.groupby('day_of_week')['fare'].mean()
axes[1, 1].bar(range(7), daily_avg.values, color='orange', edgecolor='black')
axes[1, 1].set_title('Precio Promedio por Dia')
axes[1, 1].set_xlabel('Dia de la Semana')
axes[1, 1].set_ylabel('Precio Promedio ($)')
axes[1, 1].set_xticks(range(7))
axes[1, 1].set_xticklabels(days, rotation=45)

# 6. Efecto de lluvia
rain_comparison = df.groupby('is_raining')['fare'].mean()
axes[1, 2].bar(['No Lluvia', 'Lluvia'], rain_comparison.values, 
               color=['skyblue', 'navy'], edgecolor='black')
axes[1, 2].set_title('Efecto de Lluvia en Precio')
axes[1, 2].set_ylabel('Precio Promedio ($)')
for i, v in enumerate(rain_comparison.values):
    axes[1, 2].text(i, v + 0.5, f'${v:.2f}', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('eda_taxi.png', dpi=300, bbox_inches='tight')
print("\n✅ Graficas guardadas en 'eda_taxi.png'")

# ============================================
# 3. PREPARACIÓN DE DATOS
# ============================================
print("\n🔧 PASO 3: Preparacion de Datos")
print("-" * 70)

# Separar features (X) y target (y)
feature_columns = ['distance_km', 'duration_min', 'hour', 'day_of_week', 
                   'passengers', 'temperature', 'is_raining']
X = df[feature_columns].values
y = df['fare'].values

print(f"📌 Shape de X: {X.shape}")
print(f"📌 Shape de y: {y.shape}")

# Split: 70% train, 15% validation, 15% test
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.176, random_state=42  # 15% del total
)

print(f"\n✅ Division de datos:")
print(f"   Training:   {len(X_train):,} muestras ({len(X_train)/len(X)*100:.1f}%)")
print(f"   Validation: {len(X_val):,} muestras ({len(X_val)/len(X)*100:.1f}%)")
print(f"   Test:       {len(X_test):,} muestras ({len(X_test)/len(X)*100:.1f}%)")

# Normalización (importante para redes neuronales)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

print(f"\n✅ Datos normalizados")
print(f"   Media antes: {X_train[:, 0].mean():.2f}")
print(f"   Media despues: {X_train_scaled[:, 0].mean():.2f}")
print(f"   Std despues: {X_train_scaled[:, 0].std():.2f}")

# ============================================
# 4. CREAR Y ENTRENAR MODELO
# ============================================
print("\n🧠 PASO 4: Creacion y Entrenamiento del Modelo")
print("-" * 70)

# Arquitectura de la red neuronal
modelo = keras.Sequential([
    # Capa de entrada + primera capa oculta
    keras.layers.Dense(128, activation='relu', input_shape=(7,)),
    keras.layers.Dropout(0.2),  # Regularizacion
    
    # Segunda capa oculta
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.2),
    
    # Tercera capa oculta
    keras.layers.Dense(32, activation='relu'),
    
    # Capa de salida (prediccion del precio)
    keras.layers.Dense(1)
])

print("\n Arquitectura del modelo:")
modelo.summary()

# Compilar modelo
modelo.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='mse',  # Mean Squared Error
    metrics=['mae']  # Mean Absolute Error
)

# Callbacks para mejorar entrenamiento
early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=20,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=10,
    verbose=1,
    min_lr=0.00001
)

print("\n🎓 Entrenando modelo...")
print("   (Esto demora  de 1 a 2 minutos)")

history = modelo.fit(
    X_train_scaled, y_train,
    validation_data=(X_val_scaled, y_val),
    epochs=200,
    batch_size=32,
    callbacks=[early_stopping, reduce_lr],
    verbose=0
)

print(f"\n✅ Entrenamiento completado en {len(history.history['loss'])} epocas")

# ============================================
# 5. EVALUACIÓN DEL MODELO
# ============================================
print("\n📊 PASO 5: Evaluación del Modelo")
print("-" * 70)

# Evaluar en conjunto de test
test_loss, test_mae = modelo.evaluate(X_test_scaled, y_test, verbose=0)
print(f"\n🎯 Resultados en Test Set:")
print(f"   MAE (Error Absoluto Medio): ${test_mae:.2f}")
print(f"   MSE (Error Cuadrático Medio): ${test_loss:.2f}")
print(f"   RMSE (Raíz del MSE): ${np.sqrt(test_loss):.2f}")

# Hacer predicciones
y_pred_train = modelo.predict(X_train_scaled, verbose=0).flatten()
y_pred_val = modelo.predict(X_val_scaled, verbose=0).flatten()
y_pred_test = modelo.predict(X_test_scaled, verbose=0).flatten()

# Calcular R² (coeficiente de determinación)
from sklearn.metrics import r2_score
r2_train = r2_score(y_train, y_pred_train)
r2_test = r2_score(y_test, y_pred_test)

print(f"\n📈 R² Score (que tan bien explica el modelo):")
print(f"   Training: {r2_train:.4f} ({r2_train*100:.2f}%)")
print(f"   Test:     {r2_test:.4f} ({r2_test*100:.2f}%)")

# Visualización de resultados
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Evaluacion del Modelo', fontsize=16, fontweight='bold')

# 1. Pérdida durante entrenamiento
axes[0, 0].plot(history.history['loss'], label='Training Loss', linewidth=2)
axes[0, 0].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
axes[0, 0].set_title('Perdida durante Entrenamiento')
axes[0, 0].set_xlabel('Epoca')
axes[0, 0].set_ylabel('MSE')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 2. MAE durante entrenamiento
axes[0, 1].plot(history.history['mae'], label='Training MAE', linewidth=2)
axes[0, 1].plot(history.history['val_mae'], label='Validation MAE', linewidth=2)
axes[0, 1].set_title('MAE durante Entrenamiento')
axes[0, 1].set_xlabel('Época')
axes[0, 1].set_ylabel('MAE ($)')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# 3. Predicciones vs Reales (Test set)
axes[1, 0].scatter(y_test, y_pred_test, alpha=0.5, s=20)
axes[1, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
                'r--', linewidth=2, label='Prediccion Perfecta')
axes[1, 0].set_title(f'Predicciones vs Reales (R²={r2_test:.3f})')
axes[1, 0].set_xlabel('Precio Real ($)')
axes[1, 0].set_ylabel('Precio Predicho ($)')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# 4. Distribución de errores
errors = y_test - y_pred_test
axes[1, 1].hist(errors, bins=50, color='coral', edgecolor='black')
axes[1, 1].axvline(0, color='red', linestyle='--', linewidth=2)
axes[1, 1].set_title(f'Distribucion de Errores (Media: ${errors.mean():.2f})')
axes[1, 1].set_xlabel('Error ($)')
axes[1, 1].set_ylabel('Frecuencia')
axes[1, 1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('model_evaluation.png', dpi=300, bbox_inches='tight')
print("\n✅ Graficas de evaluacion guardadas en 'model_evaluation.png'")

# ============================================
# 6. PREDICCIONES EN CASOS REALES
# ============================================
print("\n🔮 PASO 6: Predicciones en Casos Reales")
print("-" * 70)

# Casos de prueba interesantes
test_cases = [
    {
        'descripcion': '🌅 Viaje corto matutino, sin trafico',
        'features': [3.5, 12, 6, 2, 1, 18, 0]  # 3.5km, 12min, 6am, miercoles, 1 pasajero, 18°C, sin lluvia
    },
    {
        'descripcion': '🚗 Viaje medio en hora pico',
        'features': [8.2, 35, 8, 4, 2, 20, 0]  # 8.2km, 35min, 8am, viernes, 2 pasajeros, 20°C, sin lluvia
    },
    {
        'descripcion': '🌧️ Viaje largo con lluvia nocturna',
        'features': [15.0, 45, 22, 5, 1, 12, 1]  # 15km, 45min, 10pm, sábado, 1 pasajero, 12°C, lluvia
    },
    {
        'descripcion': '🏙️ Viaje muy largo de fin de semana',
        'features': [25.0, 60, 14, 6, 4, 25, 0]  # 25km, 60min, 2pm, domingo, 4 pasajeros, 25°C, sin lluvia
    }
]

for i, case in enumerate(test_cases, 1):
    features_scaled = scaler.transform([case['features']])
    precio_predicho = modelo.predict(features_scaled, verbose=0)[0][0]
    
    print(f"\n{i}. {case['descripcion']}")
    print(f"   📍 Distancia: {case['features'][0]:.1f} km")
    print(f"   ⏱️  Duración: {case['features'][1]:.0f} min")
    print(f"   🕐 Hora: {case['features'][2]:02d}:00")
    print(f"   💵 Precio predicho: ${precio_predicho:.2f}")

# ============================================
# 7. GUARDAR MODELO
# ============================================
print("\n💾 PASO 7: Guardando Modelo")
print("-" * 70)

modelo.save('taxi_fare_model.keras')
print("✅ Modelo guardado como 'taxi_fare_model.keras'")

# Guardar el scaler también
import pickle
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("✅ Scaler guardado como 'scaler.pkl'")

# ============================================
# RESUMEN FINAL
# ============================================
print("\n" + "=" * 70)
print("📊 RESUMEN DEL PROYECTO")
print("=" * 70)
print(f"""
✅ Dataset: {len(df):,} viajes analizados
✅ Features: {len(feature_columns)} variables de entrada
✅ Modelo: Red neuronal con 3 capas ocultas (128-64-32 neuronas)
✅ Performance:
   - MAE en Test: ${test_mae:.2f} (error promedio)
   - R² Score: {r2_test:.3f} (explica {r2_test*100:.1f}% de la varianza)
   - RMSE: ${np.sqrt(test_loss):.2f}

📁 Archivos generados:
   - eda_taxi.png (analisis exploratorio)
   - model_evaluation.png (evaluación del modelo)
   - taxi_fare_model.keras (modelo entrenado)
   - scaler.pkl (normalizador de datos)

🎯 Conclusión: El modelo puede predecir tarifas con un error de ~${test_mae:.2f}
""")

plt.show()