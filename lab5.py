import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from mpl_toolkits.mplot3d import Axes3D  
from matplotlib.animation import FuncAnimation 
import seaborn as sns 

sns.set_theme(style="whitegrid")

def f(x):
    return x[0]**2 + 2*x[1]**2 + 2*x[0] + 0.3 * np.arctan(x[0]*x[1])

def grad_f(x):
    denom = 1 + (x[0]*x[1])**2
    df_dx1 = 2*x[0] + 2 + 0.3 * x[1] / denom
    df_dx2 = 4*x[1] + 0.3 * x[0] / denom
    return np.array([df_dx1, df_dx2])

def steepest_descent(initial_x, target_epsilon, max_iters_sd):
    x_k_sd = initial_x.copy()
    trajectory_sd = [x_k_sd.copy()]
    f_vals_sd = [f(x_k_sd)]
    grad_norms_sd = [np.linalg.norm(grad_f(x_k_sd), np.inf)]

    print(f"\n--- Запуск для epsilon = {target_epsilon:.0e} ---")
    for k_iter_sd in range(max_iters_sd):
        g_sd = grad_f(x_k_sd)
        current_grad_norm_sd = np.linalg.norm(g_sd, np.inf)
        
        if k_iter_sd > 0:
            grad_norms_sd.append(current_grad_norm_sd)

        if current_grad_norm_sd < target_epsilon:
            print(f"Остановка на итерации {k_iter_sd+1}: норма градиента ({current_grad_norm_sd:.2e}) < epsilon ({target_epsilon:.1e})")
            break
        
        res_sd = minimize_scalar(lambda t: f(x_k_sd - t * g_sd))
        t_star_sd = res_sd.x

        if t_star_sd < 0:
            print(f"Предупреждение: на итерации {k_iter_sd+1} получен отрицательный шаг t_star = {t_star_sd:.4f}.")
        
        x_k_sd = x_k_sd - t_star_sd * g_sd
        trajectory_sd.append(x_k_sd.copy())
        f_vals_sd.append(f(x_k_sd))
    else:
        print(f"Достигнуто максимальное количество итераций ({max_iters_sd})")
        if len(grad_norms_sd) == len(trajectory_sd) - 1:
             grad_norms_sd.append(np.linalg.norm(grad_f(x_k_sd), np.inf))

    trajectory_sd = np.array(trajectory_sd)
    x_final_sd = trajectory_sd[-1]
    f_final_sd = f_vals_sd[-1]
    num_iterations_sd = len(trajectory_sd) - 1
    final_grad_norm_sd = grad_norms_sd[-1]

    print(f"Результат для epsilon = {target_epsilon:.0e}:")
    print(f"  Координаты x_min = ({x_final_sd[0]:.6f}, {x_final_sd[1]:.6f})")
    print(f"  Значение f(x_min) = {f_final_sd:.6f}")
    print(f"  Количество итераций: {num_iterations_sd}")
    print(f"  Норма градиента: {final_grad_norm_sd:.2e}")

    return target_epsilon, num_iterations_sd, x_final_sd, f_final_sd, trajectory_sd, f_vals_sd, grad_norms_sd

# --- Основной скрипт ---
x_initial = np.array([-1.0, 0.0])
max_iters_main = 100
epsilon_values = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]

results_table_data = []
all_trajectories = []

for eps in epsilon_values:
    current_eps, num_iters, x_f, f_val, traj, f_v, g_n = steepest_descent(x_initial, eps, max_iters_main)
    results_table_data.append({
        "epsilon": current_eps,
        "iterations": num_iters,
        "x_final": x_f,
        "f_final": f_val
    })
    all_trajectories.append({
        "epsilon": eps,
        "trajectory": traj,
        "f_vals": f_v,
        "grad_norms": g_n
    })

print("\n--- Сводная таблица результатов ---")
header = f"| {'Epsilon':<10} | {'Итераций':<10} | {'x_min[0]':<12} | {'x_min[1]':<12} | {'f(x_min)':<15} |"
print(header)
print("|" + "-" * (len(header) - 2) + "|")
for res in results_table_data:
    row = f"| {res['epsilon']:.0e}{' '*(10-len(f'{res['epsilon']:.0e}'))} | " \
          f"{res['iterations']:<10} | " \
          f"{res['x_final'][0]:<12.8f} | " \
          f"{res['x_final'][1]:<12.8f} | " \
          f"{res['f_final']:<15.12f} |"
    print(row)
print("|" + "-" * (len(header) - 2) + "|")

# --- Визуализация ---
# Подготовка данных для графиков
x_min_plot = min([traj["trajectory"][:,0].min() for traj in all_trajectories]) - 1.0
x_max_plot = max([traj["trajectory"][:,0].max() for traj in all_trajectories]) + 1.0
y_min_plot = min([traj["trajectory"][:,1].min() for traj in all_trajectories]) - 1.0
y_max_plot = max([traj["trajectory"][:,1].max() for traj in all_trajectories]) + 1.0

X, Y = np.meshgrid(np.linspace(x_min_plot, x_max_plot, 400),
                   np.linspace(y_min_plot, y_max_plot, 400))
Z = f([X,Y])

# 2D график
plt.figure(figsize=(12, 8))
contours = plt.contour(X, Y, Z, levels=40, cmap='viridis')
plt.clabel(contours, inline=True, fontsize=8)

colors = plt.cm.rainbow(np.linspace(0, 1, len(all_trajectories)))
for traj, color in zip(all_trajectories, colors):
    x_coords = traj["trajectory"][:,0]
    y_coords = traj["trajectory"][:,1]
    plt.plot(x_coords, y_coords, marker='o', linestyle='-', color=color, 
             label=f'ε={traj["epsilon"]:.0e}', markersize=3)

plt.scatter(x_initial[0], x_initial[1], s=100, color='blue', label='Старт $x^0$', zorder=5)
plt.title('Метод наискорейшего спуска: траектории для разных ε')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.axis('equal')
plt.tight_layout()

# 3D график
fig_3d = plt.figure(figsize=(12, 9))
ax_3d = fig_3d.add_subplot(111, projection='3d')
ax_3d.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.6, rcount=100, ccount=100)

for traj, color in zip(all_trajectories, colors):
    x_coords = traj["trajectory"][:,0]
    y_coords = traj["trajectory"][:,1]
    z_coords = traj["f_vals"]
    ax_3d.plot(x_coords, y_coords, z_coords, marker='o', color=color, 
               label=f'ε={traj["epsilon"]:.0e}', markersize=3)

ax_3d.scatter(x_initial[0], x_initial[1], f(x_initial), s=100, color='blue', 
              label='Старт $x^0$', depthshade=True)
ax_3d.set_title('3D Визуализация траекторий для разных ε')
ax_3d.set_xlabel('$x_1$')
ax_3d.set_ylabel('$x_2$')
ax_3d.set_zlabel('$f(x_1, x_2)$')
ax_3d.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax_3d.view_init(elev=30, azim=120)
plt.tight_layout()

# Анимация для последнего значения epsilon
last_traj = all_trajectories[-1]
fig_anim, ax_anim = plt.subplots(figsize=(10, 7))
ax_anim.contour(X, Y, Z, levels=40, cmap='viridis', alpha=0.7)
ax_anim.set_title(f'Анимация спуска (ε={last_traj["epsilon"]:.0e})')
ax_anim.set_xlabel('$x_1$')
ax_anim.set_ylabel('$x_2$')
ax_anim.axis('equal')
ax_anim.set_xlim(x_min_plot, x_max_plot)
ax_anim.set_ylim(y_min_plot, y_max_plot)
ax_anim.axis('equal')

x_coords = last_traj["trajectory"][:,0]
y_coords = last_traj["trajectory"][:,1]

line_anim, = ax_anim.plot([], [], 'o-', color='r', markersize=5, linewidth=2, label='Траектория')
start_marker_anim = ax_anim.scatter(x_initial[0], x_initial[1], s=100, color='blue', 
                                   label='Старт $x^0$', zorder=5)
current_point_marker, = ax_anim.plot([], [], marker='o', markersize=8, color='magenta', zorder=6)
iter_text_anim = ax_anim.text(0.02, 0.95, '', transform=ax_anim.transAxes, fontsize=12, 
                             bbox=dict(facecolor='white', alpha=0.7, pad=0.2))
ax_anim.legend(loc='upper right')

def init_animation():
    line_anim.set_data([], [])
    current_point_marker.set_data([], [])
    iter_text_anim.set_text('')
    return line_anim, current_point_marker, iter_text_anim

def update_animation(frame):
    line_anim.set_data(x_coords[:frame+1], y_coords[:frame+1])
    current_point_marker.set_data([x_coords[frame]], [y_coords[frame]])
    iter_text_anim.set_text(f'Итерация: {frame}')
    if frame == len(last_traj["trajectory"]) - 1:
        ax_anim.scatter(x_coords[-1], y_coords[-1], s=100, color='red', marker='X', 
                       label='Конец $x_{final}$', zorder=5)
    return line_anim, current_point_marker, iter_text_anim

num_frames = len(last_traj["trajectory"])
if num_frames > 0:
    animation = FuncAnimation(fig_anim, update_animation,
                              frames=num_frames, init_func=init_animation,
                              blit=True, interval=300, repeat=False)
    try:
        gif_path = f'steepest_descent_animation_eps{last_traj["epsilon"]:.0e}.gif'
        animation.save(gif_path, writer='pillow', fps=max(1, min(5, num_frames // 2 if num_frames > 1 else 1)))
        print(f"\nGIF анимация сохранена в: {gif_path}")
    except Exception as e:
        print(f"\nНе удалось сохранить GIF анимацию: {e}")

plt.show()

