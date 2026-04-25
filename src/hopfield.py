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

T_values = [0, 10, 20, 30, 40, 50]



# ============================================
# Funciones JIT para acelerar Hopfield
# ============================================

@njit
def hopfield_local_field(T_matrix, V):
    """Calcula el campo local h = T @ s donde s = 2V - 1."""
    N = len(V)
    h = np.zeros(N)
    for i in range(N):
        for j in range(N):
            h[i] += T_matrix[i, j] * (2 * V[j] - 1)
    return h

@njit
def hopfield_overlap(V, pattern):
    """Calcula el solapamiento m = (1/N) * sum_i s_i * s_pattern_i."""
    N = len(V)
    m = 0.0
    for i in range(N):
        s_i = 2 * V[i] - 1
        s_p = 2 * pattern[i] - 1
        m += s_i * s_p
    return m / N

@njit
def hopfield_mc_step(T_matrix, V, T_temp, N):
    """
    Un paso de Monte Carlo: N actualizaciones asíncronas.
    """
    for _ in range(N):
        # Elegir neurona aleatoria
        i = np.random.randint(0, N)
        
        # Calcular campo local solo para neurona i
        h_i = 0.0
        for j in range(N):
            h_i += T_matrix[i, j] * (2 * V[j] - 1)
        
        # Actualización Glauber
        if T_temp < 1e-10:
            V[i] = 1 if h_i > 0 else 0
        else:
            prob = 1.0 / (1.0 + np.exp(-2.0 * h_i / T_temp))
            V[i] = 1 if np.random.random() < prob else 0
    
    return V

@njit
def hopfield_simulate_jit(T_matrix, V0, pattern, T_temp, n_steps, measure_every):
    """
    Simula la red de Hopfield con numba.
    """
    N = len(V0)
    V = V0.copy()
    
    n_measures = n_steps // measure_every + 1
    overlaps = np.zeros(n_measures)
    times = np.zeros(n_measures)
    
    measure_idx = 0
    
    for step in range(n_steps):
        # Paso de Monte Carlo
        V = hopfield_mc_step(T_matrix, V, T_temp, N)
        
        # Medir
        if step % measure_every == 0:
            overlaps[measure_idx] = hopfield_overlap(V, pattern)
            times[measure_idx] = step
            measure_idx += 1
    
    return times[:measure_idx], overlaps[:measure_idx], V

@njit(parallel=True)
def hopfield_compute_m_vs_T(T_matrix, pattern, T_values, n_trials, n_steps, n_equilib, noise_level):
    """
    Calcula m(T) paralelizando sobre temperaturas y trials.
    """
    n_temps = len(T_values)
    N = len(pattern)
    
    m_values = np.zeros(n_temps)
    m_std = np.zeros(n_temps)
    
    measure_every = 10
    
    for t_idx in prange(n_temps):
        T_temp = T_values[t_idx]
        m_trials = np.zeros(n_trials)
        
        for trial in range(n_trials):
            # Inicializar cerca del patrón con ruido
            V0 = pattern.copy()
            for i in range(N):
                if np.random.random() < noise_level:
                    V0[i] = 1 - V0[i]
            
            # Simular
            times, overlaps, V_final = hopfield_simulate_jit(
                T_matrix, V0, pattern, T_temp, n_steps, measure_every
            )
            
            # Promediar después de equilibración
            start_idx = n_equilib // measure_every
            if start_idx < len(overlaps):
                m_eq = np.mean(np.abs(overlaps[start_idx:]))
            else:
                m_eq = np.abs(overlaps[-1])
            
            m_trials[trial] = m_eq
        
        m_values[t_idx] = np.mean(m_trials)
        m_std[t_idx] = np.std(m_trials)
    
    return m_values, m_std


class HopfieldNetwork:
    """
    Red de Hopfield con temperatura finita (acelerada con numba).
    """
    
    def __init__(self, N=100):
        self.N = N
        self.patterns = []
        self.T = None
    
    def store_patterns(self, patterns):
        """Almacena patrones usando la regla de Hebb."""
        self.patterns = [np.array(p, dtype=np.int64) for p in patterns]
        
        self.T = np.zeros((self.N, self.N))
        
        for p in self.patterns:
            s = 2 * p - 1
            self.T += np.outer(s, s)
        
        self.T /= self.N
        np.fill_diagonal(self.T, 0)
    
    def simulate(self, V0, T_temp, n_steps=10000, measure_every=100):
        """Simula usando función JIT."""
        V0_int = np.array(V0, dtype=np.int64)
        pattern = np.array(self.patterns[0], dtype=np.int64)
        
        times, overlaps, V_final = hopfield_simulate_jit(
            self.T, V0_int, pattern, T_temp, n_steps, measure_every
        )
        return times, overlaps, V_final
    
    def overlap(self, V, pattern_idx=0):
        """Calcula solapamiento."""
        return hopfield_overlap(
            np.array(V, dtype=np.int64), 
            np.array(self.patterns[pattern_idx], dtype=np.int64)
        )


# Crear red
N = 200
hopfield = HopfieldNetwork(N=N)

# Crear un patrón aleatorio
np.random.seed(42)
pattern = (np.random.random(N) > 0.5).astype(np.int64)

# Almacenar el patrón
hopfield.store_patterns([pattern])

print(f"Red de Hopfield con N = {N} neuronas")
print(f"Patrón almacenado: {np.sum(pattern)} neuronas activas ({100*np.mean(pattern):.1f}%)")
print("Funciones numba compiladas y listas.")

# Calcular curva m(T) - versión paralela
T_values = np.linspace(0.1, 2.0, 30)

n_trials = 20
n_steps = 2000
n_equilib = 1000
noise_level = 0.1

print("Calculando curva m(T) con numba + parallel...")

# Usar función paralelizada
m_values, m_std = hopfield_compute_m_vs_T(
    hopfield.T, 
    pattern, 
    T_values, 
    n_trials, 
    n_steps, 
    n_equilib, 
    noise_level
)

print("Cálculo completado.")

# Plot
fig, ax = plt.subplots(figsize=(10, 6))

ax.errorbar(T_values, m_values, yerr=m_std, fmt='bo-', capsize=3, 
            markersize=6, label='Simulación')

# Predicción teórica (campo medio): m = tanh(m/T)
T_theory = np.linspace(0.01, 2.0, 200)
m_theory = []
for T in T_theory:
    if T < 1:
        m = 0.99
        for _ in range(100):
            m = np.tanh(m / T)
        m_theory.append(m)
    else:
        m_theory.append(0)

ax.plot(T_theory, m_theory, 'r-', lw=2, label='Teoría (campo medio)')
ax.axvline(1.0, color='gray', linestyle='--', label=r'$T_c = 1$')

ax.set_xlabel('Temperatura T')
ax.set_ylabel('Solapamiento |m|')
ax.set_title(f'Transición de fase en la red de Hopfield (N = {N}, 1 patrón)')
ax.legend()
ax.set_xlim(0, 2)
ax.set_ylim(0, 1.1)

plt.tight_layout()
plt.show()

print(f"\nFigura 15: Curva de magnetización m(T) del modelo de Hopfield.")
print(f"Modelo: N={N} neuronas, 1 patrón aleatorio, {n_trials} realizaciones, numba + parallel.")
print(f"Se observa transición de fase de segundo orden en T_c ≈ 1.")

# Visualizar la dinámica de recuperación de memoria
fig = plt.figure(figsize=(15, 10))

T_examples = [0.1, 0.7, 1.5]

# Crear grid: 2 filas superiores para m(t), 3 filas inferiores para imágenes
gs = fig.add_gridspec(5, 3, height_ratios=[2, 0.3, 1, 1, 1], hspace=0.4, wspace=0.3)

# Reformatear patrón como imagen
side = int(np.sqrt(N))
if side * side != N:
    side = 14
    pattern_img = pattern[:side*side].reshape(side, side)
else:
    pattern_img = pattern[:side*side].reshape(side, side)

for col, T_temp in enumerate(T_examples):
    # Inicializar con versión corrupta del patrón
    np.random.seed(col + 100)  # Semilla diferente para cada T
    V0 = pattern.copy()
    flip_mask = np.random.random(N) < 0.3
    V0[flip_mask] = 1 - V0[flip_mask]
    
    # Simular
    times, overlaps, V_final = hopfield.simulate(V0, T_temp, n_steps=500, measure_every=1)
    
    # Panel superior: evolución del solapamiento
    ax_top = fig.add_subplot(gs[0, col])
    ax_top.plot(times, overlaps, 'b-', lw=1)
    ax_top.axhline(1, color='g', linestyle='--', alpha=0.5)
    ax_top.axhline(0, color='gray', linestyle=':', alpha=0.5)
    ax_top.set_xlabel('Paso MC')
    ax_top.set_ylabel('m')
    ax_top.set_title(f'T = {T_temp}', fontsize=12, fontweight='bold')
    ax_top.set_ylim(-0.2, 1.1)
    ax_top.grid(True, alpha=0.3)
    
    # Preparar imágenes
    if side * side == N:
        final_img = V_final[:side*side].reshape(side, side)
    else:
        final_img = V_final[:side*side].reshape(side, side)
    
    diff_img = (pattern_img != final_img).astype(float)
    error = np.mean(pattern[:side*side] != V_final[:side*side])
    
    # Fila 2: Patrón original
    ax_pat = fig.add_subplot(gs[2, col])
    im1 = ax_pat.imshow(pattern_img, cmap='gray_r', vmin=0, vmax=1)
    ax_pat.set_title('Patrón original', fontsize=10)
    ax_pat.axis('off')
    
    # Fila 3: Estado final
    ax_fin = fig.add_subplot(gs[3, col])
    im2 = ax_fin.imshow(final_img, cmap='gray_r', vmin=0, vmax=1)
    ax_fin.set_title('Estado final', fontsize=10)
    ax_fin.axis('off')
    
    # Fila 4: Diferencia (errores en rojo)
    ax_diff = fig.add_subplot(gs[4, col])
    # Crear imagen RGB: negro donde coinciden, rojo donde difieren
    diff_rgb = np.zeros((side, side, 3))
    diff_rgb[:, :, 0] = diff_img  # Canal rojo para errores
    im3 = ax_diff.imshow(diff_rgb)
    ax_diff.set_title(f'Diferencia (Error: {100*error:.1f}%)', fontsize=10)
    ax_diff.axis('off')

plt.suptitle('Recuperación de memoria en la red de Hopfield', fontsize=14, fontweight='bold', y=0.98)
plt.tight_layout()
plt.show()

print("\nFigura 16: Dinámica de recuperación de memoria a diferentes temperaturas.")
print("T = 0.3 (< Tc): Recuperación perfecta, m ≈ 1, error ≈ 0%.")
print("T = 0.7 (< Tc): Recuperación con fluctuaciones, error pequeño.")
print("T = 1.5 (> Tc): No hay recuperación, m fluctúa cerca de 0, error ≈ 50%.")
