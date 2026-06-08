# Machine Learning Algorithms

Учебный проект с реализацией классических алгоритмов машинного обучения на чистом Python (NumPy). Каждый алгоритм реализован с нуля — без использования готовых реализаций из sklearn — и сопровождается интерактивным Jupyter-ноутбуком с визуализацией.

> Проект продолжает развиваться: планируется добавление новых алгоритмов и датасетов.

---

## Структура проекта

```
Machine_Learning_algorithms/
├── src/                        # Реализации алгоритмов
│   ├── perceptron.py           # Перцептрон Розенблатта
│   ├── adaline_gd.py           # Adaline (пакетный градиентный спуск)
│   ├── adaline_sgd.py          # Adaline (стохастический градиентный спуск)
│   └── utils.py                # Вспомогательные функции (визуализация)
├── notebooks/                  # Демонстрационные ноутбуки
│   ├── 01_perceptron_demo.ipynb
│   ├── 02_adaline_demo.ipynb
│   └── 03_adaline_sgd_demo.ipynb
├── data/                       # Датасеты
│   ├── iris.data               # Датасет ирисов Фишера
│   ├── penguins.data           # Датасет пингвинов Палмера
│   └── penguins.csv
└── requirements.txt
```

---

## Реализованные алгоритмы

| Алгоритм | Файл | Ноутбук |
|---|---|---|
| Перцептрон | `src/perceptron.py` | `01_perceptron_demo.ipynb` |
| Adaline (GD) | `src/adaline_gd.py` | `02_adaline_demo.ipynb` |
| Adaline (SGD) | `src/adaline_sgd.py` | `03_adaline_sgd_demo.ipynb` |

### Перцептрон (`Perceptron`)

Бинарный классификатор с пороговой функцией активации. Обновляет веса по одному примеру за раз, сходится только на линейно разделимых данных.

### Adaline GD (`AdalineGD`)

Адаптивный линейный нейрон с функцией потерь MSE и **пакетным** градиентным спуском. Обновляет веса сразу по всей обучающей выборке за эпоху. Требует стандартизации признаков.

### Adaline SGD (`AdalineSGD`)

Та же модель, но с **стохастическим** градиентным спуском: веса обновляются по одному примеру за раз. Поддерживает `partial_fit` для онлайн-обучения. Перемешивает данные перед каждой эпохой.

---

## Датасеты

- **Ирисы Фишера** — бинарная классификация Setosa vs Versicolor по длине чашелистника и лепестка
- **Пингвины Палмера** — бинарная классификация Adelie vs Chinstrap по длине и глубине клюва  
  Источник: [Kaggle](https://www.kaggle.com/datasets/satyajeetrai/palmer-penguins-dataset-for-eda)

---

## Установка окружения

### 1. Клонирование репозитория

```bash
git clone <url>
cd Machine_Learning_algorithms
```

### 2. Создание виртуального окружения

```bash
python -m venv venv
```

Активация:

```bash
# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Установка зависимостей

```bash
pip install -r requirements.txt
```

### 4. Запуск Jupyter

```bash
pip install jupyter
jupyter notebook
```

Откройте любой ноутбук из папки `notebooks/` и запустите ячейки по порядку.

---

## Зависимости

| Пакет | Назначение |
|---|---|
| `numpy` | Векторные вычисления, реализация алгоритмов |
| `pandas` | Загрузка и предобработка данных |
| `matplotlib` | Визуализация данных и границ решения |
| `scikit-learn` | Базовая модель (LogisticRegression) для сравнения |
| `scipy` | Научные вычисления |
