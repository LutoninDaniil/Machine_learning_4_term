import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, StackingRegressor
from sklearn.neural_network import MLPRegressor
from catboost import CatBoostRegressor

print("Загрузка и предобработка данных")
# Загружаем очищенный датасет
df = pd.read_csv('ai4i2020_clean.csv')

# Целевая переменная для регрессии — Температура процесса
target_col = 'Process temperature [K]'

# Выделяем признаки, которые использовались в ЛР2 (5 ключевых числовых факторов)
features = ['Air temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]', 'torque_tool_interaction']

X = df[features]
y = df[target_col]

# Разбиваем на обучающую и тестовую выборки (80% на 20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Обучение 6 моделей (внутри Pipeline с масштабированием)")

# Чтобы в Web-приложении не масштабировать данные вручную, упакуем scaler и модели в единый Pipeline
# ML1: Полиномиальная регрессия
ml1 = Pipeline([('scaler', StandardScaler()), ('poly', PolynomialFeatures(degree=2)), ('model', Ridge(alpha=1.0))])
ml1.fit(X_train, y_train)
print("ML1 (Poly Ridge) обучена. R2:", round(ml1.score(X_test, y_test), 4))

# ML2: Градиентный бустинг Sklearn
ml2 = Pipeline([('scaler', StandardScaler()), ('model', GradientBoostingRegressor(n_estimators=100, random_state=42))])
ml2.fit(X_train, y_train)
print("ML2 (Gradient Boosting) обучена. R2:", round(ml2.score(X_test, y_test), 4))

# ML3: Продвинутый бустинг CatBoost
ml3 = Pipeline([('scaler', StandardScaler()), ('model', CatBoostRegressor(iterations=200, learning_rate=0.1, random_state=42, verbose=0))])
ml3.fit(X_train, y_train)
print("ML3 (CatBoost) обучена. R2:", round(ml3.score(X_test, y_test), 4))

# ML4: Случайный лес (Бэггинг)
ml4 = Pipeline([('scaler', StandardScaler()), ('model', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))])
ml4.fit(X_train, y_train)
print("ML4 (Random Forest) обучена. R2:", round(ml4.score(X_test, y_test), 4))

# ML5: Стэкинг (Объединяем Ridge и Random Forest, мета-модель — Ridge)
estimators = [('ridge', Ridge(alpha=1.0)), ('rf', RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1))]
ml5 = Pipeline([('scaler', StandardScaler()), ('model', StackingRegressor(estimators=estimators, final_estimator=Ridge()))])
ml5.fit(X_train, y_train)
print("ML5 (Stacking) обучена. R2:", round(ml5.score(X_test, y_test), 4))

# ML6: Нейросеть (Многослойный перцептрон)
ml6 = Pipeline([('scaler', StandardScaler()), ('model', MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42))])
ml6.fit(X_train, y_train)
print("ML6 (Neural Network MLP) обучена. R2:", round(ml6.score(X_test, y_test), 4))

print("Сериализация моделей на жесткий диск")
# Сохраняем все пайплайны в файлы .pkl с помощью модуля pickle
with open('ml1_poly_ridge.pkl', 'wb') as f: pickle.dump(ml1, f)
with open('ml2_gradient_boosting.pkl', 'wb') as f: pickle.dump(ml2, f)
with open('ml3_catboost.pkl', 'wb') as f: pickle.dump(ml3, f)
with open('ml4_random_forest.pkl', 'wb') as f: pickle.dump(ml4, f)
with open('ml5_stacking.pkl', 'wb') as f: pickle.dump(ml5, f)
with open('ml6_nn_mlp.pkl', 'wb') as f: pickle.dump(ml6, f)

print("Все модели сохранены в корень папки проекта.")