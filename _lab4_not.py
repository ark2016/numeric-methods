import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------------
# 1. Определение функций системы и их частных производных
# --------------------------------------------------------------------------------
def f1(x, y):
    # f1(x,y) = cos(x-1) + y - 0.8
    return np.cos(x - 1) + y - 0.8

def f2(x, y):
    # f2(x,y) = x - cos(y) - 2
    return x - np.cos(y) - 2

def df1_dx(x, y):
    # d/dx [cos(x-1)] = -sin(x-1)
    return -np.sin(x - 1)

def df1_dy(x, y):
    return 1

def df2_dx(x, y):
    return 1

def df2_dy(x, y):
    # d/dy[-cos(y)] = sin(y)
    return np.sin(y)

# --------------------------------------------------------------------------------
# 2. Функция для итераций метода Ньютона
# --------------------------------------------------------------------------------
def newton_method_2d(x0, y0, eps=0.01, max_iter=100):
    """
    Решает систему f1(x,y)=0, f2(x,y)=0 методом Ньютона.
    
    Параметры:
        x0, y0 : начальное приближение
        eps : критерий остановки (точность)
        max_iter : максимум итераций
    
    Возвращает:
        (x, y, k) - найденное решение и количество итераций.
    """
    x, y = x0, y0

    for k in range(1, max_iter + 1):
        # Вычисляем значения функций
        f1_val = f1(x, y)
        f2_val = f2(x, y)
        
        # Вычисляем частные производные
        a11 = df1_dx(x, y)
        a12 = df1_dy(x, y)
        a21 = df2_dx(x, y)
        a22 = df2_dy(x, y)
        
        # Определитель Якобиана
        J = a11 * a22 - a12 * a21
        if abs(J) < 1e-14:
            print("Определитель Якобиана слишком мал. Метод может не сойтись.")
            return x, y, k

        # Решаем систему: J * (delta_x, delta_y)^T = (f1, f2)^T
        # Вычисляем поправки:
        delta_x = ( f1_val * a22 - f2_val * a12 ) / J
        delta_y = ( f2_val * a11 - f1_val * a21 ) / J

        # Обновление переменных
        x_new = x - delta_x
        y_new = y - delta_y

        # Проверка условия остановки
        if np.sqrt((x_new - x)**2 + (y_new - y)**2) < eps:
            return x_new, y_new, k

        x, y = x_new, y_new
    
    # Если не сошлось за max_iter итераций
    return x, y, max_iter

# --------------------------------------------------------------------------------
# 3. Графический анализ для выбора начального приближения
# --------------------------------------------------------------------------------
# Построим кривые:
# Для f1(x,y)=0: y = 0.8 - cos(x-1)
x_vals = np.linspace(0, 4, 400)
y1 = 0.8 - np.cos(x_vals - 1)

# Для f2(x,y)=0: x = 2 + cos(y)
y_vals = np.linspace(-2, 2, 400)
x2 = 2 + np.cos(y_vals)

plt.figure(figsize=(7, 6))
plt.plot(x_vals, y1, label=r'$y = 0.8 - \cos(x-1)$')
plt.plot(x2, y_vals, label=r'$x = 2 + \cos(y)$')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Графический анализ системы\n'
          r'$\cos(x-1)+y=0.8,\quad x-\cos(y)=2$')
plt.legend()
plt.grid(True)
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.show()

# --------------------------------------------------------------------------------
# 4. Запуск метода Ньютона с выбранным начальным приближением
# --------------------------------------------------------------------------------
# Из графика можно оценить приближение. Предположим, оно выглядит как (x0, y0) = (2.5, 0.2)
x0, y0 = 2.643, 0.867

solution_x, solution_y, iters = newton_method_2d(x0, y0, eps=0.01, max_iter=50)
print(f"Найденное решение: x = {solution_x:.4f}, y = {solution_y:.4f}")
print(f"Число итераций: {iters}")
print("Проверка подстановкой:")
print(f" f1(x,y) = {f1(solution_x, solution_y):.5f}")
print(f" f2(x,y) = {f2(solution_x, solution_y):.5f}")
