import numpy as np
import matplotlib.pyplot as plt

# Параметры варианта 17
A = 1
B = 1
C = -7
D = 4

# Заданная функция f(x)
def f(x):
    return A*x**3 + B*x**2 + C*x + D

# Производная f'(x) (нужна для метода Ньютона)
def f_prime(x):
    return 3*A*x**2 + 2*B*x - 7  

# Метод дихотомии
def bisection_method(a, b, eps=0.001, max_iter=1000):
    """
    Находит корень уравнения f(x)=0 на отрезке [a, b],
    используя метод деления отрезка пополам с точностью eps.
    Возвращает (root, итерации).
    """
    if f(a) * f(b) > 0:
        raise ValueError("На концах отрезка знаки функции совпадают. Перепроверьте [a, b].")
    
    iteration = 0
    while (b - a) / 2 > eps and iteration < max_iter:
        iteration += 1
        m = (a + b) / 2.0
        if f(a) * f(m) <= 0:
            b = m
        else:
            a = m
    
    # Возвращаем середину финального отрезка
    return (a + b) / 2.0, iteration

# Метод Ньютона
import numpy as np

def newton_method(x0, eps=0.001, max_iter=1000):
    """
    Находит корень уравнения f(x)=0, используя метод Ньютона,
    начиная с начального приближения x0 и точности eps.
    Возвращает (root, итерации).
    """
    iteration = 0
    x_prev = None   # Будем хранить предыдущее приближение
    x_n = x0

    for i in range(max_iter):
        iteration += 1
        fx = f(x_n)
        fpx = f_prime(x_n)

        if abs(fpx) < 1e-14:
            raise ValueError("Производная близка к нулю. Метод Ньютона может не сойтись.")

        # Вычисление следующего приближения по формуле Ньютона
        x_next = x_n - fx / fpx

        # Если уже было предыдущее приближение, проверяем дополнительное условие:
        if x_prev is not None:
            # Функция np.sign возвращает -1, 0 или 1 в зависимости от знака разности
            if f(x_n) * f(x_n + np.sign(x_n - x_prev) * eps) < 0:
                return x_n, iteration

        # Альтернативно можно дополнительно проверить изменение x:
        if abs(x_next - x_n) < eps:
            return x_next, iteration

        # Обновляем переменные для следующей итерации
        x_prev = x_n
        x_n = x_next

    # Если решение не найдено за max_iter итераций, возвращаем последнее приближение
    return x_n, iteration


# Шаг 1. Ищем интервалы смены знака на разумном отрезке
# Попробуем пробежать отрезок [-5, 5] шагом 1, чтобы найти отрезки со сменой знака.
roots_intervals = []
x_points = np.arange(-5, 6, 1)  
for i in range(len(x_points) - 1):
    x_left = x_points[i]
    x_right = x_points[i+1]
    if f(x_left)*f(x_right) < 0:
        roots_intervals.append((x_left, x_right))

print("Интервалы со сменой знака:", roots_intervals)

# Шаг 2. Для каждого отрезка применим два метода
found_roots_bisection = []
found_roots_newton = []
for (a, b) in roots_intervals:
    # 2.1. Метод дихотомии
    try:
        root_b, iter_b = bisection_method(a, b, eps=0.001)
        found_roots_bisection.append((root_b, iter_b))
    except ValueError as e:
        print(f"Проблема в методе дихотомии на отрезке [{a}, {b}]:", e)
    
    # 2.2. Метод Ньютона (возьмём x0 серединой отрезка)
    x0 = (a + b) / 2.0
    try:
        root_n, iter_n = newton_method(x0, eps=0.001)
        found_roots_newton.append((root_n, iter_n))
    except ValueError as e:
        print(f"Проблема в методе Ньютона при x0={x0}:", e)

# Выведем результаты
print("\n=== Результаты метода дихотомии ===")
for i, (r, it) in enumerate(found_roots_bisection):
    print(f"Корень {i+1}: x = {r:.5f}, итераций = {it}")

print("\n=== Результаты метода Ньютона ===")
for i, (r, it) in enumerate(found_roots_newton):
    print(f"Корень {i+1}: x = {r:.5f}, итераций = {it}")

print("\n=== Результаты WolframAlfa ===")
print(f"Корень 1 = -3.402679")
print(f"Корень 2 = 0.683969")
print(f"Корень 3 = 1.718710")

# Шаг 3. Построим график функции и отметим найденные корни
x_plot = np.linspace(-5, 5, 400)
y_plot = f(x_plot)

plt.axhline(0, color='black', linewidth=0.8)  # Линия y=0
plt.plot(x_plot, y_plot, label='f(x) = x^3 + x^2 - 7x + 4', color='blue')

# Отметим корни, найденные дихотомией (синим) и Ньютоном (красным)
for i, (r, _) in enumerate(found_roots_bisection):
    plt.plot(r, f(r), 'bo', label='_nolegend_' if i > 0 else 'Корни (дихотомия)')
for i, (r, _) in enumerate(found_roots_newton):
    plt.plot(r, f(r), 'ro', label='_nolegend_' if i > 0 else 'Корни (Ньютон)')

plt.legend()
plt.title("Поиск корней f(x)=x^3+x^2-7x+4")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.grid(True)
plt.show()
