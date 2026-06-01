import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Настройка конфигурации страницы
st.set_page_config(page_title="РГР: Разработка Web-приложения (дашборда) для инференса (вывода) моделей ML и анализа данных", layout="wide")

# Кэширование для быстрой загрузки моделей
@st.cache_resource
def load_ml_model(model_name):
    filename = f"{model_name}.pkl"
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)
    return None

# Кэширование для загрузки данных
@st.cache_data
def load_dataset():
    if os.path.exists("ai4i2020_clean.csv"):
        return pd.read_csv("ai4i2020_clean.csv")
    return None

df = load_dataset()

# Боковая панель навигации
st.sidebar.title("Управление РГР")
page = st.sidebar.radio("Выберите веб-страницу:", [
    "Страница 1: О разработчике",
    "Страница 2: О наборе данных",
    "Страница 3: Визуализация зависимостей",
    "Страница 4: Прогнозы"
])

# Страница 1
if page == "Страница 1: О разработчике" or page.startswith("Страница 1"):
    st.title("Информация о разработчике")
    st.markdown("---")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        if os.path.exists("photo.jpg"):
            st.image("photo.jpg", width=250)
            
    with col2:
        st.subheader("Карточка студента")
        st.markdown("""
        * **ФИО:** Лутонин Даниил Михайлович
        * **Номер учебной группы:** МО-241
        * **Тема РГР:** Разработка Web-приложения (дашборда) для инференса (вывода) моделей ML и анализа данных.
        """)

# Страница 2
elif page.startswith("Страница 2"):
    st.title("Информация о наборе данных")
    st.markdown("---")
    
    st.subheader("1. Предметная область")
    st.write("""
    В рамках работы исследуется датасет **AI4I2020** (Прогнозное обслуживание оборудования). 
    Набор данных имитирует параметры работы промышленного фрезерного станка. 
    **Целевой признак:** `Process temperature [K]` — внутренняя температура процесса обработки в Кельвинах, 
    которую необходимо точно предсказывать для предотвращения аварийных ситуаций.
    """)
    
    st.subheader("2. Описание входных признаков")
    st.markdown("""
    * **Air temperature [K]** — температура окружающего воздуха (в Кельвинах).
    * **Rotational speed [rpm]** — скорость вращения шпинделя станка в минуту.
    * **Torque [Nm]** — крутящий момент, прикладываемый к инструменту (в Ньютон-метрах).
    * **Tool wear [min]** — текущий износ режущего инструмента в минутах процесса обработки.
    * **torque_tool_interaction** — производный признак взаимодействия, рассчитываемый как `Torque * Tool wear`.
    """)
    
    st.subheader("3. Особенности предобработки данных и EDA")
    st.write("""
    - Были удалены неинформативные признаки: текстовый `Product ID` и порядковый номер `UDI`.
    - Исключен категориальный признак `Type` и бинарные флаги отказов (`Machine failure`, `TWF` и др.).
    - Обучение моделей производится на масштабированных признаках посредством `StandardScaler` во избежание доминирования крупных шкал.
    """)

# Страница 3
elif page.startswith("Страница 3"):
    st.title("Визуализация зависимостей в наборе данных")
    st.markdown("---")
    
    if df is None:
        st.error("Файл 'ai4i2020_clean.csv' не найден. Поместите его в рабочую директорию!")
    else:
        st.subheader("Реализовано 4 различных вида визуализации (Matplotlib / Seaborn):")
        
        # Визуализация 1: Распределение таргета
        st.markdown("#### 1. Распределение целевой переменной (Process temperature)")
        fig1, ax1 = plt.subplots(figsize=(10, 3.5))
        sns.histplot(df['Process temperature [K]'], kde=True, color='royalblue', ax=ax1)
        ax1.set_title("Гистограмма распределения температуры процесса")
        st.pyplot(fig1)
        st.markdown("---")
        
        # Визуализация 2: Корреляционная матрица
        st.markdown("#### 2. Матрица корреляции признаков (Heatmap)")
        fig2, ax2 = plt.subplots(figsize=(7, 5))
        cols = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]', 'torque_tool_interaction']
        sns.heatmap(df[cols].corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax2)
        ax2.set_title("Тепловая карта корреляционных связей")
        st.pyplot(fig2)
        st.markdown("---")
        
        # Визуализация 3: Диаграмма рассеяния
        st.markdown("#### 3. Диаграмма рассеяния: Air Temperature vs Process Temperature")
        fig3, ax3 = plt.subplots(figsize=(10, 4))
        # Берём случайную подвыборку точек, чтобы график не перегружался
        sns.scatterplot(data=df.sample(800, random_state=42), x='Air temperature [K]', y='Process temperature [K]', color='darkorange', alpha=0.7, ax=ax3)
        ax3.set_title("Скаттерплот зависимости температур")
        st.pyplot(fig3)
        st.markdown("---")
        
        # Визуализация 4: Ящик с усами
        st.markdown("#### 4. Распределение крутящего момента (Boxplot)")
        fig4, ax4 = plt.subplots(figsize=(10, 2.5))
        sns.boxplot(x=df['Torque [Nm]'], color='mediumseagreen', ax=ax4)
        ax4.set_title("Ящик с усами для Крутящего момента (Torque)")
        st.pyplot(fig4)

# Страница 4
elif page.startswith("Страница 4"):
    st.title("Предсказание моделей машинного обучения")
    st.markdown("---")
    
    # 1. Выбор модели пользователем
    model_choice = st.selectbox("Выберите модель для выполнения прогноза:", [
        "ml1_poly_ridge (Классическая модель: Полиномиальная Ridge регрессия)",
        "ml2_gradient_boosting (Ансамбль: Градиентный бустинг)",
        "ml3_catboost (Продвинутый градиентный бустинг CatBoost)",
        "ml4_random_forest (Ансамбль: Случайный Лес / Бэггинг)",
        "ml5_stacking (Ансамбль: Стэкинг моделей)",
        "ml6_nn_mlp (Глубокая полносвязная нейросеть MLP)"
    ])
    
    model_id = model_choice.split(" ")[0]
    loaded_model = load_ml_model(model_id)
    
    if loaded_model is None:
        st.error(f"Файл '{model_id}.pkl' отсутствует. Сначала запустите скрипт save_models.py!")
    else:
        st.success(f"Модель {model_id} успешно загружена и готова к работе.")
        
    st.markdown("### Ввод параметров вручную:")
    
    # Создаем форму для ввода с валидацией и единицами измерения
    col_a, col_b = st.columns(2)
    with col_a:
        user_air_temp = st.number_input("Air temperature [K] (Температура воздуха в Кельвинах):", min_value=200.0, max_value=400.0, value=300.0, step=0.1)
        user_rot_speed = st.number_input("Rotational speed [rpm] (Скорость вращения, об/мин):", min_value=100, max_value=6000, value=1500, step=1)
    with col_b:
        user_torque = st.number_input("Torque [Nm] (Крутящий момент в Ньютон-метрах):", min_value=0.0, max_value=150.0, value=50.0, step=0.1)
        user_tool_wear = st.number_input("Tool wear [min] (Износ инструмента в минутах):", min_value=0, max_value=500, value=10, step=1)
        
    # Автоматический интеллектуальный расчет производного признака
    user_interaction = float(user_torque * user_tool_wear)
    st.info(f"Рассчитанный признак взаимодействия (torque_tool_interaction): **{user_interaction:.2f}**")
    
    # Кнопка запуска предсказания
    if st.button("✅ Выполнить прогноз температуры процесса"):
        if loaded_model is not None:
            # Формируем DataFrame с точным наименованием колонок
            input_features = pd.DataFrame([{
                'Air temperature [K]': user_air_temp,
                'Rotational speed [rpm]': user_rot_speed,
                'Torque [Nm]': user_torque,
                'Tool wear [min]': user_tool_wear,
                'torque_tool_interaction': user_interaction
            }])
            
            # Получаем предсказание
            predicted_value = loaded_model.predict(input_features)[0]
            
            # Понятная интерпретация для конечного пользователя
            st.markdown(f"### 📊 Результат прогноза: **{predicted_value:.2f} K** (градусов Кельвина)")
        else:
            st.error("Невозможно сделать прогноз: модель не загружена.")
            
    st.markdown("---")
    st.markdown("### Загрузка файла (*.csv):")
    
    uploaded_csv = st.file_uploader("Загрузите файл формата .csv для массового расчета предсказаний:", type=["csv"])
    if uploaded_csv is not None:
        batch_df = pd.read_csv(uploaded_csv)
        st.write("Исходный файл (первые строки):")
        st.dataframe(batch_df.head())
        
        required_cols = ['Air temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]', 'torque_tool_interaction']
        
        # Проверка валидности структуры загруженного файла
        if all(col in batch_df.columns for col in required_cols):
            if st.button("✅ Вычислить предикты для всей таблицы"):
                if loaded_model is not None:
                    predictions_array = loaded_model.predict(batch_df[required_cols])
                    result_df = batch_df.copy()
                    result_df['Predicted Process temperature [K]'] = predictions_array
                    
                    st.success("Массовый расчет успешно завершен.")
                    st.dataframe(result_df)
                else:
                    st.error("Модель не инициализирована.")
        else:
            st.error(f"Ошибка валидации. Убедитесь, что файл содержит все обязательные столбцы: {required_cols}")
