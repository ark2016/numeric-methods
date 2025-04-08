import numpy as np
import matplotlib.pyplot as plt

'''
y'' - 2y' + 2y = 2x
y(0) = 0
y'(0) = 1

Система:
y1' = y2
y2' = 2x + 2y1 - 2y2

Точное решение:
y(x) = 1 + x - e^x * cos(x) + e^x * sin(x)
'''

# Параметры задачи
x0 = 0
x_end = 1
y0 = np.array([0.0, 1.0])
h = 0.1
eps = 0.001
p = 4  # порядок метода Рунге-Кутты

# Правая часть
def f(x, y):
    y1, y2 = y
    return np.array([y2, 2*x + 2*y1 - 2*y2])

# Точное решение
def exact_solution(x):
    return 1 + x - np.exp(x) * np.cos(x) + np.exp(x) * np.sin(x)

# Шаг Рунге-Кутты 4-го порядка
def runge_kutta_step(x, y, h):
    k1 = f(x, y)
    k2 = f(x + h/2, y + h * k1 / 2)
    k3 = f(x + h/2, y + h * k2 / 2)
    k4 = f(x + h, y + h * k3)
    return y + (h / 6) * (k1 + 2*k2 + 2*k3 + k4)

# Основной адаптивный цикл
x_vals = [x0]
y_vals = [y0]
x = x0
y = y0
runge_errors = []

while x < x_end - 1e-10:
    if x + h > x_end:
        h = x_end - x  # последний шаг
    
    # Оценка по правилу Рунге
    y1 = runge_kutta_step(x, y, h) # шаг h
    y2_half = runge_kutta_step(x, y, h/2) # шаг h/2
    y2 = runge_kutta_step(x + h/2, y2_half, h/2)  # второй шаг h/2

    err = np.linalg.norm(y1 - y2) / (2**p - 1)
    runge_errors.append(err)

    if err <= eps:
        # шаг принят
        x += h
        y = y2  # более точное приближение
        x_vals.append(x)
        y_vals.append(y)

    # Автоматический подбор нового шага
    h_opt = h * (eps / err)**(1 / (p + 1))
    h = 0.9 * h_opt

# Массивы результатов
x_vals = np.array(x_vals)
y_vals = np.array(y_vals)

# Таблица сравнения
print("x\tЧисленное y\tТочное y\tПогрешность\tпогрешность по правилу Рунге")
for xi, yi, e_runge in zip(x_vals, y_vals, runge_errors):
    y_exact = exact_solution(xi)
    error = abs(yi[0] - y_exact)
    print(f"{xi:.2f}\t{yi[0]:.6f}\t{y_exact:.6f}\t{error:.6f}\t{e_runge:.6f}")

# График
plt.plot(x_vals, y_vals[:, 0], 'o-', label='Рунге-Кутта (адаптивный шаг)')
plt.plot(x_vals, exact_solution(x_vals), 'r--', label='Точное решение')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Рунге-Кутта с контролем ошибки (по правилу Рунге)')
plt.grid(True)
plt.legend()
plt.show()
