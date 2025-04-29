import numpy as np
import matplotlib.pyplot as plt

'''
y'' - 2y' + 2y = 2x
y(0) = 0
y'(0) = 1

Система ОДУ первого порядка:
y1' = y2
y2' = 2x - 2*y1 + 2*y2  
Точное решение:
y(x) = 1 + x - exp(x)*cos(x) + exp(x)*sin(x)
'''

# --- Параметры задачи ---
x0 = 0.0
x_end = 1.0
y0 = np.array([0.0, 1.0]) # [y(0), y'(0)]
h = 0.1 # Начальный шаг 
eps = 1e-6 # Желаемая точность локальной ошибки 
p = 4  # Порядок метода Рунге-Кутты 

# --- Правая часть системы ОДУ ---
def f(x, y):
    y1, y2 = y
    return np.array([y2, 2*x - 2*y1 + 2*y2])

# --- Точное решение ---
def exact_solution(x):
    return 1 + x - np.exp(x) * np.cos(x) + np.exp(x) * np.sin(x) 

# --- Шаг метода Рунге-Кутты 4-го порядка ---
def runge_kutta_step(x, y, h, f_func):
    k1 = f_func(x, y)
    k2 = f_func(x + h/2, y + h * k1 / 2)
    k3 = f_func(x + h/2, y + h * k2 / 2)
    k4 = f_func(x + h, y + h * k3)
    return y + (h / 6) * (k1 + 2*k2 + 2*k3 + k4)

# --- Основной адаптивный цикл с правилом Рунге и уточнением Ричардсона ---
x_vals = [x0]
y_vals = [y0]
step_sizes = []
runge_errors_est = [] # Оценки локальной ошибки (до экстраполяции)
actual_errors = [0.0] # Фактическая ошибка в узлах

x = x0
y = y0.copy() # Используем копию, чтобы не изменять y0

max_steps = 10000 # Ограничение на число шагов
step_count = 0

while x < x_end - 1e-10 and step_count < max_steps:
    step_count += 1
    
    # Убедимся, что не перешагнем конец интервала
    if x + h > x_end:
        h = x_end - x

    # --- Оценка погрешности по правилу Рунге ---
    y1 = runge_kutta_step(x, y, h, f)
    y2_half = runge_kutta_step(x, y, h/2, f)
    y2 = runge_kutta_step(x + h/2, y2_half, h/2, f)

    # --- Расчет оценки локальной ошибки ---
    error_diff = y2 - y1
    local_error_estimate = np.max(np.abs(error_diff)) / (2**p - 1)


    # --- Адаптация шага ---
    if local_error_estimate <= eps or h < 1e-12: 
        x += h
        
        # --- Уточнение по Ричардсону ---
        # Уточненное значение = Более точное значение (y2) + Оценка его ошибки
        # Оценка ошибки y2 ≈ (y2 - y1) / (2^p - 1)
        richardson_correction = error_diff / (2**p - 1)
        y = y2 + richardson_correction 

        x_vals.append(x)
        y_vals.append(y.copy()) 
        step_sizes.append(h)
        runge_errors_est.append(local_error_estimate) 
        actual_errors.append(np.abs(y[0] - exact_solution(x))) 

        if local_error_estimate > 1e-15: 
            fac = (eps / local_error_estimate)**(1.0 / (p + 1.0))
        else:
             fac = 2.0 

        h_new = 0.9 * h * min(max(fac, 0.1), 2.0) 

    else:
        if local_error_estimate > 1e-15:
             fac = (eps / local_error_estimate)**(1.0 / (p + 1.0))
        else:
             fac = 0.5 
        h_new = 0.9 * h * min(max(fac, 0.1), 1.0) 
        
    h = h_new 
    
    if h < 1e-14:
        print(f"Warning: Step size became extremely small ({h}) at x={x}. Stopping.")
        break

if step_count >= max_steps:
    print(f"Warning: Maximum number of steps ({max_steps}) reached. Calculation may be incomplete.")

# --- Обработка и вывод результатов ---
x_vals = np.array(x_vals)
y_vals = np.array(y_vals)
step_sizes = np.array(step_sizes)
runge_errors_est = np.array(runge_errors_est)
actual_errors = np.array(actual_errors)

print("\n--- Результаты ---")
print(f"Количество принятых шагов: {len(x_vals) - 1}")
print(f"Минимальный размер шага: {np.min(step_sizes) if len(step_sizes) > 0 else 'N/A'}")
print(f"Максимальный размер шага: {np.max(step_sizes) if len(step_sizes) > 0 else 'N/A'}")
print(f"Максимальная оцененная локальная ошибка: {np.max(runge_errors_est) if len(runge_errors_est) > 0 else 'N/A'}")
print(f"Максимальная фактическая ошибка (в узлах): {np.max(actual_errors) if len(actual_errors) > 0 else 'N/A'}")
print(f"Фактическая ошибка в конечной точке x={x_vals[-1]:.4f}: {actual_errors[-1]:.2e}")


# --- Таблица сравнения ---
print("\nТаблица: x | Численное y | Точное y | Факт. ошибка (y) | Оценка лок. ошибки")
print("-" * 70)
# Выводим только несколько точек для наглядности + последнюю
indices_to_print = np.linspace(0, len(x_vals)-1, num=10, dtype=int)
if len(x_vals)-1 not in indices_to_print: # Убедимся, что последняя точка есть
     indices_to_print = np.append(indices_to_print, len(x_vals)-1)
     indices_to_print = np.unique(indices_to_print) # Удалить дубликаты, если есть

for i in indices_to_print:
    xi = x_vals[i]
    yi = y_vals[i]
    y_exact = exact_solution(xi)
    # Оценка ошибки относится к шагу *до* текущей точки (кроме первой)
    err_est_str = f"{runge_errors_est[i-1]:.2e}" if i > 0 else "N/A"
    print(f"{xi:6.3f} | {yi[0]:11.8f} | {y_exact:11.8f} | {actual_errors[i]:17.2e} | {err_est_str}")

# --- Графики ---
plt.figure(figsize=(12, 10))

# График решения
plt.subplot(3, 1, 1)
plt.plot(x_vals, y_vals[:, 0], 'bo-', label='Рунге-Кутта-Ричардсон (адаптивный)', markersize=4, linewidth=1)
# Для сравнения можно построить точное решение в большем количестве точек
x_dense = np.linspace(x0, x_end, 200)
plt.plot(x_dense, exact_solution(x_dense), 'r--', label='Точное решение')
plt.xlabel('x')
plt.ylabel('y(x)')
plt.title(f'Адаптивный РК4 с уточнением Ричардсона (eps = {eps:.1e})')
plt.grid(True)
plt.legend()

# График фактической ошибки
plt.subplot(3, 1, 2)
plt.plot(x_vals, actual_errors, 'mo-', label='Фактическая ошибка |y_num - y_exact|', markersize=4)
plt.xlabel('x')
plt.ylabel('Абсолютная ошибка')
plt.title('Фактическая ошибка в узлах сетки')
plt.yscale('log')
plt.grid(True)
plt.legend()

# График изменения размера шага
plt.subplot(3, 1, 3)
# Строим шаги напротив интервалов, которые они покрывают (x_vals[1:] соответствует step_sizes)
plt.plot(x_vals[1:], step_sizes, '.-', label='Размер шага h(x)')
plt.xlabel('x (конец интервала шага)')
plt.ylabel('Шаг h')
plt.title('Адаптивный шаг')
plt.yscale('log')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
