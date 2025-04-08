import numpy as np
import matplotlib.pyplot as plt

'''
y" - 2y' + 2y = 2x
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
y0 = np.array([0.0, 1.0])  # [y1, y2]
h = 0.1  # шаг

# Правая часть системы
def f(x, y):
    y1, y2 = y
    dy1 = y2
    dy2 = 2*x + 2*y1 - 2*y2
    return np.array([dy1, dy2])

# Точное аналитическое решение
def exact_solution(x):
    return 1 + x - np.exp(x) * np.cos(x) + np.exp(x) * np.sin(x)

# Метод Рунге-Кутта 4-го порядка
def runge_kutta_step(x, y, h):
    k1 = f(x, y)
    k2 = f(x + h/2, y + h * k1 / 2)
    k3 = f(x + h/2, y + h * k2 / 2)
    k4 = f(x + h, y + h * k3)
    return y + (h / 6) * (k1 + 2*k2 + 2*k3 + k4)

# Основной цикл
x_vals = [x0]
y_vals = [y0]
x = x0
y = y0

while x < x_end - 1e-10:
    if x + h > x_end:
        h = x_end - x  # последний шаг
    y = runge_kutta_step(x, y, h)
    x += h
    x_vals.append(x)
    y_vals.append(y)

x_vals = np.array(x_vals)
y_vals = np.array(y_vals)

# Табличный вывод сравнения
print("x\tЧисленное y\tТочное y\tПогрешность")
for xi, yi in zip(x_vals, y_vals):
    y_exact = exact_solution(xi)
    error = abs(yi[0] - y_exact)
    print(f"{xi:.2f}\t{yi[0]:.6f}\t{y_exact:.6f}\t{error:.2e}")

# Построение графика
plt.plot(x_vals, y_vals[:, 0], 'o-', label='Рунге-Кутта (численно)')
plt.plot(x_vals, exact_solution(x_vals), 'r--', label='Точное решение')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Сравнение: численное и точное решение')
plt.grid(True)
plt.legend()
plt.show()
