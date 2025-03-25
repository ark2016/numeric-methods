import numpy as np

def f(x):
    return np.log(x)**2 / x

def rectangle_method(a, b, n):
    h = (b - a) / n
    return h * sum(f(a + (k + 0.5) * h) for k in range(n))

def trapezoidal_method(a, b, n):
    h = (b - a) / n
    return h * (sum(f(a + k * h) for k in range(1, n)) + (f(a) + f(b)) / 2)


def simpson_method(a, b, n):
    if n % 2 != 0:
        raise ValueError("Число разбиений n должно быть чётным для метода "
        "Симпсона.")

    h = (b - a) / n
    return h / 3 * (f(a) + f(b) + sum(2 * f(a + k * h) if k % 2 == 0 
                                      else 4 * f(a + k * h) 
                                      for k in range(1, n)))

def runge_method(method, a, b, n, p):
    return (method(a, b, n) - method(a, b, 2*n)) / (2**p - 1)

def adaptive_integration(method, a, b, eps, p):
    n = 2
    while True:
        if abs(runge_method(method, a, b, n, p)) < eps:
            return method(a, b, n), n
        n *= 2

def main():
    a = 1 / np.e # левая граница
    b = np.e # правая граница
    eps = 0.001 # требуемая точность

    # Метод прямоугольников:
    rect_val, rect_n = adaptive_integration(rectangle_method, a, b, eps, 2)

    # Метод трапеций:
    trap_val, trap_n = adaptive_integration(trapezoidal_method, a, b, eps, 2)

    # Метод Симпсона:
    simp_val, simp_n = adaptive_integration(simpson_method, a, b, eps, 4)

    # Метод Рунге:
    runge_rect = runge_method(rectangle_method, a, b, 128, 2)
    runge_trap = runge_method(trapezoidal_method, a, b, 128, 2)
    runge_simp = runge_method(simpson_method, a, b, 32, 4)

    # Вывод результатов
    print(f"{'':<15} {'метод прямоугольников':<20} {'метод трапеций':<20}" 
          f"{'метод Симпсона':<20}")
    print(f"{'n':<15} {rect_n:<20} {trap_n:<20} {simp_n:<20}")
    print(f"{'I(n)':<15} {rect_val:<20.6f} {trap_val:<20.6f}" 
          f"{simp_val:<20.6f}")
    print(f"{'R':<15} {runge_rect:<20.6f} {runge_trap:<20.6f}"
          f"{runge_simp:<20.6f}")
    print(f"{'I(n) + R':<15} {rect_val + runge_rect:<20.6f} "
          f"{trap_val + runge_trap:<20.6f}{simp_val + runge_simp:<20.6f}")

if __name__ == "__main__":
    main()
    