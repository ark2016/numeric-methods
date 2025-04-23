import numpy as np
import math
import matplotlib.pyplot as plt

# --- Определение задачи ---

# Определим систему ОДУ первого порядка: y' = f(x, y)
# y — это вектор [y1, y2] = [y, y']
def f(x, y):
    y1, y2 = y
    dy1_dx = y2
    dy2_dx = 2.0 * x + 2.0 * y1 - 2.0 * y2
    return np.array([dy1_dx, dy2_dx])

# Точное решение y(x) (это y1)
def exact_solution(x):
    # Обработка как скалярных, так и массивных входов
    if isinstance(x, (np.ndarray, list)):
        x = np.asarray(x)
        return 1.0 + x - np.exp(x) * np.cos(x) + np.exp(x) * np.sin(x)
    else:
        return 1.0 + x - math.exp(x) * math.cos(x) + math.exp(x) * math.sin(x)

# --- Один шаг метода Рунге-Кутты 4-го порядка ---
def rk4_step(func, x, y, h):
    """Выполняет один шаг метода Рунге-Кутты 4-го порядка."""
    k1 = h * func(x, y)
    k2 = h * func(x + 0.5*h, y + 0.5*k1)
    k3 = h * func(x + 0.5*h, y + 0.5*k2)
    k4 = h * func(x + h, y + k3)
    return y + (k1 + 2*k2 + 2*k3 + k4) / 6.0

# --- Адаптивный решатель методом РК4 ---
def solve_ode_adaptive_rk4(func, y0, x_span, h_init, tol, h_min=1e-6, h_max=0.5, safety=0.9):
    """
    Решает dy/dx = func(x, y) методом РК4 с адаптивным шагом.

    Аргументы:
        func: функция, определяющая систему ОДУ f(x, y).
        y0: начальное условие y(x_start).
        x_span: кортеж (x_start, x_end).
        h_init: начальное приближение шага.
        tol: желаемая локальная погрешность.
        h_min: минимально допустимый шаг.
        h_max: максимально допустимый шаг.
        safety: коэффициент безопасности для коррекции шага.

    Возвращает:
        Кортеж: (x_values, y_values, runge_error_estimates)
    """
    x_start, x_end = x_span
    x = x_start
    y = np.asarray(y0, dtype=float)
    h = h_init

    x_vals = [x]
    y_vals = [y.copy()]
    runge_errors = [0.0] # Оценка погрешности для принятого шага

    p = 4 # Порядок метода РК4
    eps = 1e-16 # Малое число, чтобы избежать деления на ноль

    while x < x_end:
        # Не выходить за границу x_end
        if x + h > x_end:
            h = x_end - x

        # Убедимся, что h не слишком мал
        h = max(h, h_min)

        while True: # Внутренний цикл для коррекции шага
            # Один шаг с размером h
            y1 = rk4_step(func, x, y, h)

            # Два шага с размером h/2
            y_mid = rk4_step(func, x, y, h / 2.0)
            y2 = rk4_step(func, x + h / 2.0, y_mid, h / 2.0)

            # Оценка ошибки по правилу Рунге (максимум по компонентам)
            error_estimate = np.max(np.abs(y1 - y2)) / (2**p - 1)

            # Вычисление масштабного коэффициента
            if error_estimate < eps:
                scale = 2.0 # Увеличиваем шаг, если ошибка очень мала
            else:
                scale = safety * (tol / error_estimate) ** (1.0 / (p + 1.0))

            # Ограничим масштаб, чтобы избежать резких изменений
            scale = min(2.0, max(0.2, scale))

            # Принимаем ли шаг?
            if error_estimate <= tol:
                x = x + h
                y = y2
                accepted_error = error_estimate
                h_new = min(h_max, max(h_min, h * scale))

                x_vals.append(x)
                y_vals.append(y.copy())
                runge_errors.append(accepted_error)

                h = h_new
                break # Переход к следующему шагу
            else:
                h_new = max(h_min, h * scale)
                h = h_new

                if h <= h_min + eps:
                    print(f"Предупреждение: шаг достиг минимума ({h_min}) при x={x}. Завершение.")
                    return np.array(x_vals), np.array(y_vals), np.array(runge_errors)

        if abs(h) < eps:
            print(f"Предупреждение: шаг стал слишком малым ({h}) при x={x}. Завершение.")
            break

    return np.array(x_vals), np.array(y_vals), np.array(runge_errors)

# --- Параметры ---
y0 = [0.0, 1.0]       # Начальные условия [y(0), y'(0)]
x_span = (0.0, 1.0)   # Интервал [x_start, x_end]
h_init = 0.1          # Начальное приближение шага
tolerance = 1e-5      # Желаемая локальная ошибка

# --- Решение ---
x_vals, y_vals, runge_errors = solve_ode_adaptive_rk4(f, y0, x_span, h_init, tolerance)

# --- Вывод результатов ---

print("x\tЧисленное y\tТочное y\t abs err\t R err est")
print("-------------------------------------------------------------------")
for i in range(len(x_vals)):
    xi = x_vals[i]
    yi = y_vals[i]
    y_num = yi[0]
    e_runge = runge_errors[i]

    y_exact_val = exact_solution(xi)
    abs_error = abs(y_num - y_exact_val)
    print(f"{xi:.4f}\t{y_num:.6f}\t{y_exact_val:.6f}\t{abs_error:.6f}\t{e_runge:.6E}")

# График решения
plt.figure(figsize=(10, 6))
plt.plot(x_vals, y_vals[:, 0], 'o-', markersize=4, linewidth=1, label='Рунге-Кутта (адаптивный шаг)')
plt.plot(x_vals, exact_solution(x_vals), 'r--', linewidth=1.5, label='Точное решение')

plt.xlabel('x')
plt.ylabel('y(x)')
plt.title('Рунге-Кутта с адаптивным шагом (Контроль ошибки по правилу Рунге)')
plt.grid(True)
plt.legend()
plt.show()

# График размеров шага
plt.figure(figsize=(10, 4))
step_sizes = np.diff(x_vals)
plt.plot(x_vals[1:], step_sizes, '.-', label='Размер шага h')
plt.xlabel('x')
plt.ylabel('h')
plt.title('Адаптивный размер шага h(x)')
plt.grid(True)
plt.yscale('log')
plt.legend()
plt.tight_layout()
plt.show()

# График y'(x)
plt.figure(figsize=(10, 6))
plt.plot(x_vals, y_vals[:, 1], 'o-', markersize=4, linewidth=1, label='Численное y\'(x)')
# Точная производная y'(x)
y_prime_exact = 1.0 + 2.0 * np.exp(x_vals) * np.sin(x_vals)
plt.plot(x_vals, y_prime_exact, 'g--', linewidth=1.5, label='Точное y\'(x)')
plt.xlabel('x')
plt.ylabel('y\'(x)')
plt.title('Сравнение производной y\'(x)')
plt.grid(True)
plt.legend()
plt.show()
