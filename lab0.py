def progonka(n, a, b, c, f, alpha0, beta0, alpha_n, beta_n):
    # Чтобы не путаться с a_0, b_0 и a_n, b_n, введём другие буквы:
    p = [0.0] * (n + 1)
    q = [0.0] * (n + 1)

    # Начальные условия "прямого хода" берем из левого края:
    # Считаем, что p[-1] = alpha0, q[-1] = beta0.
    # Логически скажем:
    p[0] = alpha0
    q[0] = beta0

    # Прямой ход для i=1..n-1
    for i in range(1, n):
        denom = b[i] + a[i] * p[i - 1]  # знаменатель
        p[i] = - c[i] / denom
        q[i] = (f[i] - a[i] * q[i - 1]) / denom

    # Теперь считаем x_n из правого граничного условия:
    #   x_n = (beta_n + alpha_n * q[n-1]) / (1 - alpha_n * p[n-1])
    x = [0.0] * (n + 1)
    x[n] = (beta_n + alpha_n * q[n - 1]) / (1 - alpha_n * p[n - 1])

    # Обратный ход: i=n-1..0
    for i in range(n - 1, -1, -1):
        x[i] = p[i] * x[i + 1] + q[i]

    return x

def init_from_matrix(matrix, f_vector):
    n = len(matrix) - 1  # Размер внутренней системы
    a = [0] * (n + 1)
    b = [0] * (n + 1)
    c = [0] * (n + 1)
    f = [0] * (n + 1)
    for i in range(1, n):  # Только внутренние уравнения
        b[i] = matrix[i][i]
        a[i] = matrix[i][i - 1]
        c[i] = matrix[i][i + 1]
        f[i] = f_vector[i]
    return a, b, c, f

def test_matrix_example():
    matrix = [
        [4, 1, 0, 0, 0],
        [1, 4, 1, 0, 0],
        [0, 1, 4, 1, 0],
        [0, 0, 1, 4, 1],
        [0, 0, 0, 1, 4]
    ]
    f_vector = [5, 6, 6, 6, 5]
    a, b, c, f = init_from_matrix(matrix, f_vector)
    alpha0, beta0 = 0, 1  # x[0] = 1
    alpha_n, beta_n = 0, 1  # x[4] = 1
    x = progonka(len(matrix) - 1, a, b, c, f, alpha0, beta0, alpha_n, beta_n)
    print("Решение системы:", x)

if __name__ == '__main__':
    test_matrix_example()
