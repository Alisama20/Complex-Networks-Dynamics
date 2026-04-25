import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint, solve_ivp
from scipy.signal import find_peaks
from numba import njit, prange

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Configuración de gráficos
plt.rcParams['figure.figsize'] = [10, 6]
plt.rcParams['font.size'] = 11
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3



# --- Definiciones de ejercicios anteriores ---

T = 100  # ms


def measure_width(V):
    """
    Calcula el ancho efectivo (desviacion estandar) de la distribucion
    de potencial 2D, tratando |V| como densidad de probabilidad.

    Returns: sigma (float) — ancho RMS en unidades de pixel
    """
    N = V.shape[0]
    x = np.arange(N)
    y = np.arange(N)
    X, Y = np.meshgrid(x, y)

    V_abs = np.abs(V)
    total = V_abs.sum()
    if total == 0:
        return 0.0
    P = V_abs / total

    x_mean = np.sum(X * P)
    y_mean = np.sum(Y * P)

    sigma2 = np.sum(((X - x_mean)**2 + (Y - y_mean)**2) * P)
    return np.sqrt(sigma2)


# Verificación numérica: difusión en red 2D con sinapsis eléctricas

def simulate_electrical_synapses_2D(N=50, T=100, dt=0.1, G_e=0.5):
    """
    Simula difusión de potencial en red 2D con sinapsis eléctricas.
    
    Ecuación: dV/dt = -G_e * sum_vecinos(V_i - V_j) = D * Laplaciano(V)
    
    Parámetros:
    -----------
    N : int
        Tamaño de la red (N x N)
    T : float
        Tiempo total de simulación
    dt : float
        Paso temporal
    G_e : float
        Conductancia de gap junction
    """
    # Inicialización: pulso gaussiano en el centro
    x = np.arange(N)
    y = np.arange(N)
    X, Y = np.meshgrid(x, y)
    
    # Condición inicial: gaussiana centrada
    V = np.exp(-((X - N//2)**2 + (Y - N//2)**2) / (2 * (N//10)**2))
    V = V * 50  # Escalar para visualización
    
    # Guardar estados
    n_steps = int(T / dt)
    snapshots = [V.copy()]
    times = [0]
    
    # Simulación (Euler explícito con Laplaciano discreto)
    save_interval = max(1, n_steps // 5)  # Evitar división por cero
    
    for step in range(n_steps):
        # Laplaciano con condiciones de frontera periódicas
        laplacian = (
            np.roll(V, 1, axis=0) + np.roll(V, -1, axis=0) +
            np.roll(V, 1, axis=1) + np.roll(V, -1, axis=1) - 4 * V
        )
        
        # Actualizar (D = G_e ya que dx = 1)
        V = V + dt * G_e * laplacian
        
        # Guardar snapshots
        if step % save_interval == 0:
            snapshots.append(V.copy())
            times.append((step + 1) * dt)
    
    return snapshots, times

# Ejecutar simulación
snapshots, times = simulate_electrical_synapses_2D(N=50, T=50, G_e=0.3)

# Visualización
fig, axes = plt.subplots(2, 3, figsize=(14, 9))
axes = axes.flatten()

vmin, vmax = 0, np.max(snapshots[0])

for i, (V, t) in enumerate(zip(snapshots, times)):
    im = axes[i].imshow(V, cmap='hot', vmin=vmin, vmax=vmax, origin='lower')
    axes[i].set_title(f't = {t:.1f}')
    axes[i].set_xlabel('x')
    axes[i].set_ylabel('y')
    plt.colorbar(im, ax=axes[i], label='V')

plt.suptitle('Difusión del potencial en red 2D con sinapsis eléctricas\n' + 
             r'$I^{syn} = G_e \sum_j (V_i - V_j) \equiv D \nabla^2 V$', fontsize=12)
plt.tight_layout()
plt.show()

print("\nFigura 11: Difusión de potencial en red 2D con sinapsis eléctricas.")
print(f"Modelo: Red regular k=4, N=50×50, G_e=0.3, T=50.")
print("La condición inicial (gaussiana) se difunde isotrópicamente, confirmando I^syn ~ D∇²V.")

# Verificación cuantitativa: una sola simulación midiendo en tiempos específicos

def simulate_and_measure(N=50, T_max=50, dt=0.1, G_e=0.3, measure_times=[0, 10, 20, 30, 40, 50]):
    """
    Simula difusión y mide el ancho en tiempos específicos.
    """
    n_steps = int(T_max / dt)
    
    # Condición inicial
    x = np.arange(N)
    y = np.arange(N)
    X, Y = np.meshgrid(x, y)
    V = np.exp(-((X - N//2)**2 + (Y - N//2)**2) / (2 * (N//10)**2)) * 50
    
    # Medir ancho inicial
    widths = [measure_width(V)]
    measured_times = [0]
    
    current_time = 0
    measure_idx = 1  # Ya medimos t=0
    
    for step in range(n_steps):
        # Laplaciano con condiciones de frontera periódicas
        laplacian = (
            np.roll(V, 1, axis=0) + np.roll(V, -1, axis=0) +
            np.roll(V, 1, axis=1) + np.roll(V, -1, axis=1) - 4 * V
        )
        V = V + dt * G_e * laplacian
        current_time += dt
        
        # Medir si alcanzamos un tiempo objetivo
        if measure_idx < len(measure_times) and current_time >= measure_times[measure_idx] - dt/2:
            widths.append(measure_width(V))
            measured_times.append(current_time)
            measure_idx += 1
    
    return np.array(measured_times), np.array(widths)

# Ejecutar
G_e = 0.3
D = G_e
T_values = [0, 10, 20, 30, 40, 50]

measured_times, widths_sim = simulate_and_measure(N=50, T_max=50, dt=0.1, G_e=G_e, measure_times=T_values)

# Predicción teórica (2D: σ² = σ₀² + 4Dt)
sigma_0 = widths_sim[0]
widths_theory = np.sqrt(sigma_0**2 + 4*D*np.array(T_values))

# Plot
fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(measured_times, widths_sim, 'bo-', markersize=10, lw=2, label='Simulación')
ax.plot(T_values, widths_theory, 'r--', lw=2, label=r'Teoría: $\sigma(t) = \sqrt{\sigma_0^2 + 4Dt}$')

ax.set_xlabel('Tiempo t')
ax.set_ylabel(r'Ancho $\sigma$ de la distribución')
ax.set_title(r'Verificación de la ecuación de difusión: $\partial V/\partial t = D \nabla^2 V$')
ax.legend()

plt.tight_layout()
plt.show()

print(f"\nFigura 12: Verificación cuantitativa de la equivalencia I^syn ↔ D∇²V.")
print(f"Parámetros: G_e = {G_e}, D = {D}, σ₀ = {sigma_0:.2f}.")
print(f"Simulación y teoría coinciden para difusión 2D.")
