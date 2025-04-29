import numpy as np
import matplotlib.pyplot as plt

'''
Задача:
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
h = 0.5 # Начальный шаг 
eps = 1e-8 
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

current_h = h

while x < x_end - 1e-10 and step_count < max_steps:
    step_count += 1
    
    # Убедимся, что не перешагнем конец интервала
    attempt_h = current_h # Размер шага для текущей 
    if x + attempt_h > x_end:
        attempt_h = x_end - x

    if attempt_h < 1e-14:
        print(f"Warning: Attempted step size became extremely small ({attempt_h:.2e}) at x={x:.4f}. Stopping.")
        break

    y1_attempt = runge_kutta_step(x, y, attempt_h, f)
    y2_half_attempt = runge_kutta_step(x, y, attempt_h/2, f)
    y2_attempt = runge_kutta_step(x + attempt_h/2, y2_half_attempt, attempt_h/2, f)

    # --- Расчет оценки локальной ошибки ---
    # Ошибка оценивается как разность между более точным (y2_attempt) и менее точным (y1_attempt) решением,
    # деленная на (2^p - 1). Это оценка ошибки для y2_attempt.
    error_diff_attempt = y2_attempt - y1_attempt
    local_error_estimate = np.max(np.abs(error_diff_attempt)) / (2**p - 1)


    # --- Адаптация шага ---
    if local_error_estimate <= eps: 
        x += attempt_h
        
        # --- Уточнение по Ричардсону ---
        # Уточненное значение = Более точное значение (y2_attempt) + Оценка его ошибки
        # Оценка ошибки y2_attempt ≈ (y2_attempt - y1_attempt) / (2^p - 1)
        richardson_correction = error_diff_attempt / (2**p - 1)
        y = y2_attempt + richardson_correction 

        x_vals.append(x)
        y_vals.append(y.copy()) 
        step_sizes.append(attempt_h) 
        runge_errors_est.append(local_error_estimate) 
        actual_errors.append(np.abs(y[0] - exact_solution(x)))

        if local_error_estimate > 1e-15: 
            fac = (eps / local_error_estimate)**(1.0 / (p + 1.0))
        else:
             fac = 2.0 # Если ошибка очень мала, можно увеличивать шаг

        current_h = 0.9 * attempt_h * min(max(fac, 0.1), 2.0) 

    else: 
        if local_error_estimate > 1e-15:
             fac = (eps / local_error_estimate)**(1.0 / (p + 1.0))
        else:
             fac = 0.5 

        current_h = 0.9 * attempt_h * min(max(fac, 0.1), 1.0) 


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
print(f"Минимальный размер принятого шага: {np.min(step_sizes) if len(step_sizes) > 0 else 'N/A'}")
print(f"Максимальный размер принятого шага: {np.max(step_sizes) if len(step_sizes) > 0 else 'N/A'}")


# --- Таблица сравнения  ---
header = f"{'x':<8} | {'Численное y':<13} | {'Точное y':<11} | {'Факт. ошибка (y)':<18} | {'Оценка лок. ошибки':<18}"
print(header)
print("-" * len(header))

# Выбираем индексы для печати: начало, конец и несколько промежуточных
num_points_in_table = 15
if len(x_vals) <= num_points_in_table:
    indices_to_print = np.arange(len(x_vals))
else:
    indices_to_print = np.unique(np.linspace(0, len(x_vals) - 1, num_points_in_table, dtype=int))


for i in indices_to_print:
    xi = x_vals[i]
    yi = y_vals[i]
    y_exact = exact_solution(xi)
    
    actual_err_i = actual_errors[i]
    err_est_str = f"{runge_errors_est[i-1]:<18.2e}" if i > 0 else f"{'N/A':<18}"
    
    print(f"{xi:<8.5f} | {yi[0]:<13.8f} | {y_exact:<11.8f} | {actual_err_i:<18.2e} | {err_est_str}")

# --- Графики ---
plt.figure(figsize=(10, 12)) 

# График решения
plt.subplot(4, 1, 1)
plt.plot(x_vals, y_vals[:, 0], 'bo-', label='Численное решение (с уточнением по Ричардсону)', markersize=4, linewidth=1)
x_dense = np.linspace(x0, x_end, 200)
plt.plot(x_dense, exact_solution(x_dense), 'r--', label='Точное решение')
plt.xlabel('x')
plt.ylabel('y(x)')
plt.grid(True)
plt.legend()

# График абсолютной ошибки
plt.subplot(4, 1, 2)
plt.plot(x_vals, actual_errors, 'mo-', label='абсолютная ошибка |y_num - y_exact|', markersize=4)
plt.xlabel('x')
plt.ylabel('Абсолютная ошибка')
plt.yscale('log')
plt.grid(True)
plt.legend()

# График оцененной погрешности(по Рунге)
plt.subplot(4, 1, 3)
plt.plot(x_vals[1:], runge_errors_est, 'cs-', label='Оценка погрешности по Рунге', markersize=4)
plt.xlabel('x (конец интервала шага)')
plt.ylabel('Оценка погрешности')
plt.yscale('log')
plt.grid(True)
plt.legend()


# График изменения размера шага
plt.subplot(4, 1, 4)
plt.plot(x_vals[1:], step_sizes, '.-', label='Размер шага h(x)')
plt.xlabel('x (конец интервала шага)')
plt.ylabel('Шаг h')
plt.yscale('log')
plt.grid(True)
plt.legend()

plt.tight_layout() 
plt.show()