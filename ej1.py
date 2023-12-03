import numpy as np
from scipy.linalg import svd
import matplotlib.pyplot as plt

# Definir las dimensiones
n = 5
d = 100

# Generar matrices aleatorias A y vector aleatorio b
A = np.random.rand(n, d)
b = np.random.rand(n)

# Definir la función objetivo F(x)
def F(x):
    Ax_minus_b = np.dot(A, x) - b
    return (np.transpose(Ax_minus_b) @ Ax_minus_b)

# Hessian para F(x)
Hessian_F = 2 * np.dot(A.T, A)

# Calcular los autovalores
eigenvalues = np.linalg.eigvals(Hessian_F)

# Generar punto de inicio aleatorio
x0 = np.random.rand(d)

# Calcular sigma_max mediante SVD
sigma_max = np.max(svd(A, compute_uv=False))

# Calcular lambda_max
lambda_max = max(eigenvalues)

# Definir el valor de delta^2
delta_squared = 1e-2 * sigma_max

# Definir el valor de s
s = 1 / lambda_max

# Definir la función objetivo F2(x)
def F2(x):
    Fx = F(x)
    norm_squared = np.linalg.norm(x, ord=2) ** 2
    return Fx + delta_squared * norm_squared

def grad_F(x):
    return 2 * np.dot(A.T, (np.dot(A, x) - b))

# Gradiente de F2(x)
def grad_F2(x):
    return 2 * np.dot(A.T, (np.dot(A, x) - b)) + 2 * delta_squared * x

# Función para el descenso de gradiente
def gradient_descent(func, grad_func, x0, s, num_iterations):
    x_history = []
    cost_history = []  # Agregar lista para almacenar valores de la función de costo

    x = x0.copy()

    for _ in range(num_iterations):
        gradient = grad_func(x)
        x = x - s * gradient
        x_history.append(x.copy())
        cost_history.append(func(x))  # Almacenar valores de la función de costo

    return x, x_history, cost_history
# Realizar el descenso de gradiente para F(x)
num_iterations = 1000
x_min_F, x_history_F, cost_history_F = gradient_descent(F, grad_F, x0, s, num_iterations)

# Realizar el descenso de gradiente para F2(x)
x_min_F2, x_history_F2, cost_history_F2 = gradient_descent(F2, grad_F2, x0, s, num_iterations)

# Obtener la solución mediante SVD
_, _, V = svd(A)
x_svd = np.dot(V.T, np.dot(np.diag(1 / eigenvalues), np.dot(V, np.dot(A.T, b))))

# Graficar la evolución del descenso de gradiente
plt.figure(figsize=(10, 6))
plt.plot(np.arange(num_iterations), [F(x) for x in x_history_F], label='F(x)')
plt.plot(np.arange(num_iterations), [F2(x) for x in x_history_F2], label='F2(x)')
plt.xlabel('Iteraciones')
plt.ylabel('Valor de la función objetivo')
plt.legend()
plt.show()

# Comparar la solución final de F(x) con SVD
x_min_F, x_history_F, cost_history_F = gradient_descent(F, grad_F, x0, s, num_iterations)
x_min_F2, x_history_F2, cost_history_F2 = gradient_descent(F2, grad_F2, x0, s, num_iterations)

# Obtener la solución mediante SVD
_, _, V = svd(A)

# Calcular la diferencia punto por punto
difference_history_F_vs_SVD = [np.linalg.norm(x_F - x_svd) for x_F in x_history_F]

difference_history_F2_vs_SVD = [np.linalg.norm(x_F2 - x_svd) for x_F2 in x_history_F2]

plt.plot(np.arange(num_iterations), difference_history_F2_vs_SVD, label='F2(x) vs SVD')
plt.plot(np.arange(num_iterations), difference_history_F_vs_SVD, label='F(x) vs SVD')

plt.xlabel('Iteraciones')
plt.ylabel('Norma de la diferencia')
plt.legend()
plt.show()


delta_squared_values = [1e-6, 1e-4, 1e-2, 1e-1]

# Realizar experimentos para diferentes valores de delta^2
for delta_squared in delta_squared_values:
    # Definir la función objetivo F2(x) con el nuevo valor de delta^2
    delta = delta_squared * sigma_max
    def F2_different_deltas(x):
        Fx = F(x)
        norm_squared = np.linalg.norm(x, ord=2) ** 2
        return Fx + delta * norm_squared

    # Realizar el descenso de gradiente para F2(x)
    _, x_history_New_F2, cost_history_New_F2 = gradient_descent(F2_different_deltas, grad_F2, x0, s, num_iterations)

    # Graficar la evolución de la función de costo para diferentes valores de delta^2
    plt.plot(np.arange(num_iterations), cost_history_New_F2, label=f'Delta = {delta_squared}')

plt.xlabel('Iteraciones')
plt.ylabel('Valor de la función objetivo (F2(x))')
plt.legend()
plt.show()
