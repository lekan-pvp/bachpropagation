import numpy as np

def sigmoid(z):
    """
    сигмоидная функция активации на чистый вход z
    """
    return 1 / (1 + np.exp(-z))

def forward_propagation(X, Y, W1, b1, W2, b2):
    """
    Вычисляет операцию прямого распространения нейроггой сети и
    возвращает результат после применения сигмоиды
    """
    net_h = np.dot(W1, X) + b1 # чистый вывод на скрытом слое
    out_h = sigmoid(net_h) # после применения сигмоиды
    net_y = np.dot(W2, out_h) + b2 # чистый вывод на уровне вывода
    out_y = sigmoid(net_y) # вактический вывод на выходном слое

    return out_h, out_y

def backward_propagation(X, Y, out_h, out_y, W2):
    """
    Вычисляет операцию обратного распространения нейронной сети и
    возвращает производную весов и смещений
    """
    l2_error = out_y - Y # актуальный выход - цель
    dW2 = np.dot(l2_error, out_h.T) # производная весов уровня 2 - это точечный продукт ошибки на уровне 2 и выводе скрытого слоя
    db2 = np.sum(l2_error, axis=1, keepdims=True) # производная смещения уровня 2 - это просто ошибка на уровне 2

    dh = np.dot(W2.T, l2_error) # вычисляем скалярное произведение весов на уровне 2 с ошибкой на уровне 2
    l1_error = np.multiply(dh, out_h * (1 - out_h)) # вычисляем ошибку на уровне 1
    dW1 = np.dot(l1_error, X.T) # производная весов уровня 1 - это скалярное умножение ошибки уровня 1 на входные данные
    db1 = np.sum(l1_error, axis=1, keepdims=True) # производная смещения уровня 1 - это просто ошибка на уровне 1

    return dW1, db1, dW2, db2

def update_parameters(W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate):
    """
    обновляет веса и смещения
    """
    W1 = W1 - learning_rate * dW1
    W2 = W2 - learning_rate * dW2
    b1 = b1 - learning_rate * db1
    b2 = b2 - learning_rate * db2

    return W1, b1, W2, b2

# Инициализация параметров
np.random.seed(42) # инициализация с тем же слуяайным числом
X = np.array([[0, 0, 1, 1], [0, 1, 0, 1]]) # вхлдной массив
Y = np.array([[0, 1, 1, 0]]) # метка вывода
n_h = 2 # количество нейронов в скрытом слое
n_x = X.shape[0] # количество нейронов во взодном слое
n_y = Y.shape[0] # количество нейронов в выходном слое
W1 = np.random.randn(n_h, n_x) # веса из входного слоя
b1 = np.zeros((n_h, 1)) # смещение в скрытом слое
W2 = np.random.randn(n_y, n_h) # веса из скрытого слоя
b2 = np.zeros((n_y, 1)) # смещение в выходном слое
num_iterations = 100000
learning_rate = 0.01

# проход прямого распределения
A1, A2 = forward_propagation(X, Y, W1, b1, W2, b2)

# проход обратного распределения
dW1, db1, dW2, db2 = backward_propagation(X, Y, A1, A2, W2)

# обновление параметров
W1, b1, W2, b2 = update_parameters(W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate)

# проход прямого распределения
A1, A2 = forward_propagation(X, Y, W1, b1, W2, b2)

# вычисление прогнозоа
pred = (A2 > 0.5) * 1
print("Predicted label:", pred) # прогнозируемое значение
