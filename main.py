import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')

def f(x: float) -> float:
    return np.log(x)**2 / x

a = 1/np.e
b = np.e

h = (b-a)/32

def print_func(a, b, h):
    print([[x, f(x)] for x in np.arange(a, b, h)])

print_func(a, b, h)

# Generate points
points = [[x, f(x)] for x in np.arange(a, b, h)]

plt.figure(figsize=(10, 6))
plt.plot([point[0] for point in points], [point[1] for point in points], marker='o')
plt.title('Plot of f(x) = (ln(x)^2) / x')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.grid(True)
plt.show()

def thomas_algorithm(a, b, c, d):
    n = len(d)
    c_prime = np.zeros(n)
    d_prime = np.zeros(n)
    c_prime[0] = c[0] / b[0]
    d_prime[0] = d[0] / b[0]

    for i in range(1, n):
        temp = b[i] - a[i] * c_prime[i - 1]
        c_prime[i] = c[i] / temp
        d_prime[i] = (d[i] - a[i] * d_prime[i - 1]) / temp

    x = np.zeros(n)
    x[-1] = d_prime[-1]

    for i in range(n - 2, -1, -1):
        x[i] = d_prime[i] - c_prime[i] * x[i + 1]

    return x

n = len(points) - 1

y = [point[1] for point in points]

a_coeffs = np.ones(n - 1)
b_coeffs = 4 * np.ones(n - 1)
c_coeffs = np.ones(n - 1)
d_coeffs = np.array([(y[i + 2] - 2 * y[i + 1] + y[i]) / h**2 for i in range(n - 1)])

c_0 = 0
c_n = 0
c_i_values = [c_0] + list(thomas_algorithm(a_coeffs, b_coeffs, c_coeffs, d_coeffs)) + [c_n]

a_i = y

b_i = [(y[i+1] - y[i]) / h - h / 3 * (c_i_values[i+1] + 2 * c_i_values[i]) for i in range(n)]

d_i = [(c_i_values[i+1] - c_i_values[i])/(3*h) for i in range(n)]

print("a_i values:", a_i)
print("b_i values:", b_i)
print("c_i values:", c_i_values)
print("d_i values:", d_i)

x_i = [a + (i - .5) * h for i in range(n + 1)]
points_2 = [[x, f(x)] for x in x_i]

print(points_2)

plt.figure(figsize=(10, 6))
plt.plot([point[0] for point in points_2], [point[1] for point in points_2], marker='o', color='red')
plt.title('Plot of f(x) = (ln(x)^2) / x')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.grid(True)
plt.show()

def evaluate_spline(x, x_i, a_i, b_i, c_i, d_i):
    i = np.searchsorted(x_i, x) - 1
    i = max(0, min(i, len(x_i) - 2))

    dx = x - x_i[i]
    value = a_i[i] + b_i[i] * dx + c_i[i] * dx**2 + d_i[i] * dx**3
    return value

x_eval = 1.5
original_value = f(x_eval)
spline_value = evaluate_spline(x_eval, x_i, a_i, b_i, c_i_values, d_i)

print(f"Original function value at x = {x_eval}: {original_value}")
print(f"Spline value at x = {x_eval}: {spline_value}")


x_eval = a
original_value = f(x_eval)
spline_value = evaluate_spline(x_eval, x_i, a_i, b_i, c_i_values, d_i)

print(f"Original function value at x = {x_eval}: {original_value}")
print(f"Spline value at x = {x_eval}: {spline_value}")

x_eval = b
original_value = f(x_eval)
spline_value = evaluate_spline(x_eval, x_i, a_i, b_i, c_i_values, d_i)

print(f"Original function value at x = {x_eval}: {original_value}")
print(f"Spline value at x = {x_eval}: {spline_value}")