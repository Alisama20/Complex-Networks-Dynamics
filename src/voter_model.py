import numpy as np
import matplotlib.pyplot as plt
from numba import njit, prange
from scipy.integrate import odeint, solve_ivp
from scipy.optimize import fsolve

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Configuración de gráficos
plt.rcParams['figure.figsize'] = [10, 6]
plt.rcParams['font.size'] = 11
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3



# ============================================
# Generadores de redes
# ============================================

@njit
def create_ring_lattice(N, k):
    """
    Crea una red en anillo regular donde cada nodo está conectado
    a sus k/2 vecinos más cercanos a cada lado.
    
    Retorna lista de adyacencia como matriz (N, k).
    """
    neighbors = np.zeros((N, k), dtype=np.int64)
    half_k = k // 2
    
    for i in range(N):
        idx = 0
        for j in range(1, half_k + 1):
            neighbors[i, idx] = (i - j) % N
            neighbors[i, idx + 1] = (i + j) % N
            idx += 2
    
    return neighbors


def create_watts_strogatz(N, k, p, seed=None):
    """
    Crea una red pequeño mundo de Watts-Strogatz.
    
    Parámetros:
    -----------
    N : int
        Número de nodos
    k : int
        Grado de cada nodo (debe ser par)
    p : float
        Probabilidad de recableado
    
    Retorna:
    --------
    adj_list : list of lists
        Lista de adyacencia
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Crear anillo regular
    adj_set = [set() for _ in range(N)]
    half_k = k // 2
    
    for i in range(N):
        for j in range(1, half_k + 1):
            adj_set[i].add((i + j) % N)
            adj_set[i].add((i - j) % N)
            adj_set[(i + j) % N].add(i)
            adj_set[(i - j) % N].add(i)
    
    # Recablear con probabilidad p
    for i in range(N):
        for j in range(1, half_k + 1):
            target = (i + j) % N
            if np.random.random() < p:
                # Elegir nuevo target
                candidates = [n for n in range(N) if n != i and n not in adj_set[i]]
                if candidates:
                    new_target = np.random.choice(candidates)
                    # Eliminar arista antigua
                    adj_set[i].discard(target)
                    adj_set[target].discard(i)
                    # Añadir arista nueva
                    adj_set[i].add(new_target)
                    adj_set[new_target].add(i)
    
    # Convertir a lista de arrays
    adj_list = [np.array(list(s), dtype=np.int64) for s in adj_set]
    
    return adj_list


def create_barabasi_albert(N, m, seed=None):
    """
    Crea una red de Barabási-Albert mediante attachment preferencial.
    
    Parámetros:
    -----------
    N : int
        Número final de nodos
    m : int
        Número de aristas que añade cada nuevo nodo
    
    Retorna:
    --------
    adj_list : list of lists
        Lista de adyacencia
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Empezar con un grafo completo de m+1 nodos
    adj_set = [set() for _ in range(N)]
    
    for i in range(m + 1):
        for j in range(i + 1, m + 1):
            adj_set[i].add(j)
            adj_set[j].add(i)
    
    # Lista de stubs para attachment preferencial
    stubs = []
    for i in range(m + 1):
        stubs.extend([i] * len(adj_set[i]))
    
    # Añadir nodos uno a uno
    for new_node in range(m + 1, N):
        # Elegir m nodos distintos con probabilidad proporcional a su grado
        targets = set()
        while len(targets) < m:
            target = stubs[np.random.randint(len(stubs))]
            if target != new_node:
                targets.add(target)
        
        # Conectar nuevo nodo
        for target in targets:
            adj_set[new_node].add(target)
            adj_set[target].add(new_node)
            stubs.append(new_node)
            stubs.append(target)
    
    # Convertir a lista de arrays
    adj_list = [np.array(list(s), dtype=np.int64) for s in adj_set]
    
    return adj_list


def adj_list_to_arrays(adj_list):
    """
    Convierte lista de adyacencia a formato compatible con numba.
    
    Retorna:
    --------
    neighbors : array (total_edges,)
        Array plano con todos los vecinos
    neighbor_ptr : array (N+1,)
        Punteros al inicio de vecinos de cada nodo
    degrees : array (N,)
        Grado de cada nodo
    """
    N = len(adj_list)
    degrees = np.array([len(adj_list[i]) for i in range(N)], dtype=np.int64)
    neighbor_ptr = np.zeros(N + 1, dtype=np.int64)
    neighbor_ptr[1:] = np.cumsum(degrees)
    
    total_edges = neighbor_ptr[-1]
    neighbors = np.zeros(total_edges, dtype=np.int64)
    
    for i in range(N):
        start = neighbor_ptr[i]
        end = neighbor_ptr[i + 1]
        neighbors[start:end] = adj_list[i]
    
    return neighbors, neighbor_ptr, degrees

print("Funciones de generación de redes definidas.")

# ============================================
# Modelo del votante con numba
# ============================================

@njit
def voter_step(states, neighbors, neighbor_ptr, degrees, N):
    """
    Un paso del modelo del votante (dinámica de nodos).
    - Elige nodo i al azar
    - Elige vecino j al azar
    - i copia opinión de j
    """
    # Elegir nodo al azar
    i = np.random.randint(0, N)
    
    # Elegir vecino al azar
    if degrees[i] > 0:
        start = neighbor_ptr[i]
        end = neighbor_ptr[i + 1]
        j_idx = np.random.randint(0, degrees[i])
        j = neighbors[start + j_idx]
        
        # Copiar opinión
        states[i] = states[j]
    
    return states


@njit
def voter_step_link(states, neighbors, neighbor_ptr, degrees, N, total_edges):
    """
    Un paso del modelo del votante (dinámica de aristas).
    - Elige arista (i,j) al azar
    - Con prob 0.5, i copia a j; sino j copia a i
    
    Esta dinámica conserva la magnetización en redes heterogéneas.
    """
    # Elegir arista al azar (elegir nodo proporcional a su grado)
    edge_idx = np.random.randint(0, total_edges)
    
    # Encontrar nodo i tal que edge_idx está en sus vecinos
    i = 0
    while neighbor_ptr[i + 1] <= edge_idx:
        i += 1
    
    j = neighbors[edge_idx]
    
    # Con probabilidad 0.5, i copia a j; sino j copia a i
    if np.random.random() < 0.5:
        states[i] = states[j]
    else:
        states[j] = states[i]
    
    return states


@njit
def compute_magnetization(states, N):
    """Calcula la magnetización m = (1/N) * sum(s_i)."""
    return np.sum(states) / N


@njit
def compute_active_interfaces(states, neighbors, neighbor_ptr, N):
    """
    Calcula la densidad de interfases activas.
    n_A = (pares con diferente opinión) / (total de pares)
    """
    n_different = 0
    n_total = 0
    
    for i in range(N):
        start = neighbor_ptr[i]
        end = neighbor_ptr[i + 1]
        for idx in range(start, end):
            j = neighbors[idx]
            if j > i:  # Contar cada par una sola vez
                n_total += 1
                if states[i] != states[j]:
                    n_different += 1
    
    if n_total == 0:
        return 0.0
    return n_different / n_total


@njit
def simulate_voter(states, neighbors, neighbor_ptr, degrees, N, n_steps, 
                   measure_every, use_link_dynamics=False):
    """
    Simula el modelo del votante.
    
    Parámetros:
    -----------
    states : array
        Estados iniciales (+1 o -1)
    n_steps : int
        Número de pasos de Monte Carlo (cada paso = N actualizaciones)
    measure_every : int
        Frecuencia de medición
    use_link_dynamics : bool
        Si True, usa dinámica de aristas (conserva magnetización)
    
    Retorna:
    --------
    times, magnetizations, interface_densities
    """
    total_edges = neighbor_ptr[-1]
    
    n_measures = n_steps // measure_every + 1
    times = np.zeros(n_measures)
    magnetizations = np.zeros(n_measures)
    interface_densities = np.zeros(n_measures)
    
    measure_idx = 0
    
    for step in range(n_steps):
        # N actualizaciones = 1 paso de Monte Carlo
        for _ in range(N):
            if use_link_dynamics:
                states = voter_step_link(states, neighbors, neighbor_ptr, 
                                         degrees, N, total_edges)
            else:
                states = voter_step(states, neighbors, neighbor_ptr, degrees, N)
        
        # Medir
        if step % measure_every == 0:
            times[measure_idx] = step
            magnetizations[measure_idx] = compute_magnetization(states, N)
            interface_densities[measure_idx] = compute_active_interfaces(
                states, neighbors, neighbor_ptr, N)
            measure_idx += 1
        
        # Comprobar si hemos llegado a consenso
        if abs(compute_magnetization(states, N)) > 0.999:
            # Rellenar el resto con el valor final
            for idx in range(measure_idx, n_measures):
                times[idx] = step + (idx - measure_idx) * measure_every
                magnetizations[idx] = magnetizations[measure_idx - 1]
                interface_densities[idx] = 0.0
            break
    
    return times[:measure_idx], magnetizations[:measure_idx], interface_densities[:measure_idx], states

print("Funciones del modelo del votante compiladas.")

# ============================================
# Simulación en red pequeño mundo
# ============================================

N = 1000
k = 4  # Grado inicial
n_steps = 10000  # Más pasos para ver el plateau
measure_every = 10

# Usar p más pequeños para ver el efecto de plateau
p_values = [0, 0.005, 0.01, 0.02]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for p in p_values:
    print(f"Simulando red WS con p = {p}...")
    
    # Crear red
    adj_list = create_watts_strogatz(N, k, p, seed=42)
    neighbors, neighbor_ptr, degrees = adj_list_to_arrays(adj_list)
    
    # Estado inicial aleatorio
    np.random.seed(123)
    states = np.random.choice([-1, 1], size=N).astype(np.int64)
    
    # Simular (sin condición de parada)
    times, mags, n_A, final_states = simulate_voter(
        states.copy(), neighbors, neighbor_ptr, degrees, N, 
        n_steps, measure_every, use_link_dynamics=False
    )
    
    # Graficar magnetización
    axes[0, 0].plot(times, mags, lw=1.5, label=f'p = {p}')
    
    # Graficar densidad de interfases
    axes[0, 1].plot(times, n_A, lw=1.5, label=f'p = {p}')

# Configurar paneles
axes[0, 0].axhline(0, color='gray', linestyle='--', alpha=0.5)
axes[0, 0].set_xlabel('Paso MC')
axes[0, 0].set_ylabel('Magnetización m')
axes[0, 0].set_title('Magnetización vs tiempo')
axes[0, 0].legend()
axes[0, 0].set_ylim(-1.1, 1.1)

axes[0, 1].set_xlabel('Paso MC')
axes[0, 1].set_ylabel('Densidad de interfases $n_A$')
axes[0, 1].set_title('Interfases activas vs tiempo')
axes[0, 1].legend()
axes[0, 1].set_yscale('log')

# ============================================
# Promedio sobre realizaciones para ver plateau
# ============================================

n_trials = 10
n_steps_avg = 8000
p_compare = [0, 0.01]
colors = ['blue', 'red']

for idx, p in enumerate(p_compare):
    print(f"Promediando {n_trials} realizaciones para p = {p}...")
    
    all_n_A = []
    all_times = []
    
    for trial in range(n_trials):
        adj_list = create_watts_strogatz(N, k, p, seed=trial)
        neighbors, neighbor_ptr, degrees = adj_list_to_arrays(adj_list)
        
        np.random.seed(trial + 1000)
        states = np.random.choice([-1, 1], size=N).astype(np.int64)
        
        times, mags, n_A, _ = simulate_voter(
            states.copy(), neighbors, neighbor_ptr, degrees, N,
            n_steps_avg, measure_every, use_link_dynamics=False
        )
        
        all_n_A.append(n_A)
        all_times.append(times)
    
    # Encontrar longitud mínima
    min_len = min(len(x) for x in all_n_A)
    all_n_A_trimmed = np.array([x[:min_len] for x in all_n_A])
    times_common = all_times[0][:min_len]
    
    # Calcular promedio y desviación
    mean_n_A = np.mean(all_n_A_trimmed, axis=0)
    std_n_A = np.std(all_n_A_trimmed, axis=0)
    
    axes[1, 0].plot(times_common, mean_n_A, color=colors[idx], lw=2, label=f'p = {p}')
    axes[1, 0].fill_between(times_common, 
                            np.maximum(mean_n_A - std_n_A, 1e-4), 
                            mean_n_A + std_n_A,
                            color=colors[idx], alpha=0.2)

# Añadir línea teórica t^{-1/2} para comparar
t_theory = np.linspace(100, 8000, 100)
n_A_theory = 0.5 * (t_theory / 100) ** (-0.5)
axes[1, 0].plot(t_theory, n_A_theory, 'k--', lw=1.5, label=r'$\sim t^{-1/2}$')

axes[1, 0].set_xlabel('Paso MC')
axes[1, 0].set_ylabel(r'$\langle n_A \rangle$')
axes[1, 0].set_title(f'Densidad de interfases promedio (N={N}, {n_trials} realizaciones)')
axes[1, 0].legend()
axes[1, 0].set_yscale('log')
axes[1, 0].set_xscale('log')

# Panel informativo
axes[1, 1].text(0.5, 0.7, 'Red pequeño mundo (Watts-Strogatz)', 
                ha='center', va='center', fontsize=14, fontweight='bold',
                transform=axes[1, 1].transAxes)
axes[1, 1].text(0.5, 0.5, f'N = {N}, k = {k}', 
                ha='center', va='center', fontsize=12,
                transform=axes[1, 1].transAxes)
axes[1, 1].text(0.5, 0.3, 'p = 0: $n_A \\sim t^{-1/2}$ (consenso)\np > 0: Plateau (estado metaestable)', 
                ha='center', va='center', fontsize=11,
                transform=axes[1, 1].transAxes)
axes[1, 1].axis('off')

plt.tight_layout()
plt.show()

print(f"\nFigura 1: Modelo del votante en red pequeño mundo (Watts-Strogatz).")
print(f"Modelo: N = {N} agentes, k = {k} vecinos iniciales, dinámica de nodos.")
print(f"Observaciones:")
print(f"  - p = 0 (anillo regular): n_A decae como t^(-1/2) hacia consenso.")
print(f"  - p > 0: Los atajos crean dominios estables (plateau en n_A).")

# ============================================
# Comparación del TIEMPO de consenso
# ============================================

N = 500  # Más pequeño para alcanzar consenso más rápido
k = 4
n_trials = 10
p_values = [0, 0.01, 0.05]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Medir tiempo de consenso (cuando |m| > 0.99)
consensus_times = {p: [] for p in p_values}

for p in p_values:
    print(f"Midiendo tiempo de consenso para p = {p}...")
    
    for trial in range(n_trials):
        adj_list = create_watts_strogatz(N, k, p, seed=trial)
        neighbors, neighbor_ptr, degrees = adj_list_to_arrays(adj_list)
        
        np.random.seed(trial + 5000)
        states = np.random.choice([-1, 1], size=N).astype(np.int64)
        
        # Simular hasta consenso o máximo de pasos
        max_steps = 50000
        step = 0
        while step < max_steps:
            for _ in range(N):
                states = voter_step(states, neighbors, neighbor_ptr, degrees, N)
            step += 1
            
            m = compute_magnetization(states, N)
            if abs(m) > 0.99:
                consensus_times[p].append(step)
                break
        else:
            consensus_times[p].append(max_steps)  # No alcanzó consenso

# Panel 1: Histograma de tiempos de consenso
for i, p in enumerate(p_values):
    times = consensus_times[p]
    axes[0].hist(times, bins=15, alpha=0.5, label=f'p = {p}, media = {np.mean(times):.0f}')

axes[0].set_xlabel('Tiempo de consenso (pasos MC)')
axes[0].set_ylabel('Frecuencia')
axes[0].set_title(f'Distribución del tiempo de consenso (N = {N})')
axes[0].legend()

# Panel 2: Tiempo medio vs p
p_range = [0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1]
mean_times = []
std_times = []

for p in p_range:
    times_p = []
    for trial in range(10):
        adj_list = create_watts_strogatz(N, k, p, seed=trial + 100)
        neighbors, neighbor_ptr, degrees = adj_list_to_arrays(adj_list)
        
        np.random.seed(trial + 6000)
        states = np.random.choice([-1, 1], size=N).astype(np.int64)
        
        max_steps = 50000
        step = 0
        while step < max_steps:
            for _ in range(N):
                states = voter_step(states, neighbors, neighbor_ptr, degrees, N)
            step += 1
            if abs(compute_magnetization(states, N)) > 0.99:
                break
        times_p.append(step)
    
    mean_times.append(np.mean(times_p))
    std_times.append(np.std(times_p))

axes[1].errorbar(p_range, mean_times, yerr=std_times, fmt='o-', capsize=5, markersize=8)
axes[1].set_xlabel('Probabilidad de recableado p')
axes[1].set_ylabel('Tiempo medio de consenso')
axes[1].set_title(f'Efecto de los atajos en el tiempo de consenso (N = {N})')
axes[1].set_xscale('log')
axes[1].axhline(mean_times[0], color='gray', linestyle='--', alpha=0.5, label=f'p=0: {mean_times[0]:.0f}')
axes[1].legend()

plt.tight_layout()
plt.show()

print(f"\nFigura 1b: Tiempo de consenso en el modelo del votante.")
print(f"Modelo: N = {N}, k = {k}. A tamaño finito, ambos casos alcanzan consenso.")
print(f"Observaciones:")
print(f"  - p = 0: Consenso lento (difusión en 1D, tiempo ~ N^2).")
print(f"  - p > 0: Los atajos ACELERAN el consenso a tamaño finito.")
print(f"  - El plateau es estable solo en el límite termodinámico (N → ∞).")

# ============================================
# Simulación en red de Barabási-Albert
# TEST DE CONSERVACIÓN DE MAGNETIZACIÓN
# ============================================

N = 500
m_BA = 2
n_steps = 1000
measure_every = 10
n_trials = 30

# Fijar magnetización inicial m(0) = 0.2 (60% +1, 40% -1)
m0_target = 0.2
n_plus = int(N * (1 + m0_target) / 2)  # Número de +1

print("Simulando con m(0) = 0.2 fijo...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

all_mags_node = []
all_mags_link = []

for trial in range(n_trials):
    adj_list = create_barabasi_albert(N, m_BA, seed=trial)
    neighbors, neighbor_ptr, degrees = adj_list_to_arrays(adj_list)
    
    # Estado inicial con m(0) = 0.2 fijo
    np.random.seed(trial + 3000)
    states = np.ones(N, dtype=np.int64) * (-1)
    plus_indices = np.random.choice(N, n_plus, replace=False)
    states[plus_indices] = 1
    
    # Dinámica de nodos
    times_n, mags_n, n_A_n, _ = simulate_voter(
        states.copy(), neighbors, neighbor_ptr, degrees, N,
        n_steps, measure_every, use_link_dynamics=False
    )
    all_mags_node.append(mags_n)
    
    # Dinámica de aristas (misma CI)
    times_l, mags_l, n_A_l, _ = simulate_voter(
        states.copy(), neighbors, neighbor_ptr, degrees, N,
        n_steps, measure_every, use_link_dynamics=True
    )
    all_mags_link.append(mags_l)

# Procesar
min_len = min(min(len(x) for x in all_mags_node), min(len(x) for x in all_mags_link))
times_common = times_n[:min_len]
mags_node = np.array([x[:min_len] for x in all_mags_node])
mags_link = np.array([x[:min_len] for x in all_mags_link])

# Panel 1: Realizaciones individuales
for i in range(min(10, n_trials)):
    axes[0, 0].plot(times_common, mags_node[i], 'b-', alpha=0.3, lw=0.8)
    axes[0, 0].plot(times_common, mags_link[i], 'r-', alpha=0.3, lw=0.8)
axes[0, 0].axhline(m0_target, color='green', linestyle='--', lw=2, label=f'm(0) = {m0_target}')
axes[0, 0].axhline(0, color='gray', linestyle=':', alpha=0.5)
axes[0, 0].plot([], [], 'b-', lw=2, label='Dinámica de nodos')
axes[0, 0].plot([], [], 'r-', lw=2, label='Dinámica de aristas')
axes[0, 0].set_xlabel('Paso MC')
axes[0, 0].set_ylabel('m')
axes[0, 0].set_title('Magnetización (realizaciones individuales)')
axes[0, 0].legend()
axes[0, 0].set_ylim(-1.1, 1.1)

# Panel 2: Magnetización promedio
mean_m_node = np.mean(mags_node, axis=0)
mean_m_link = np.mean(mags_link, axis=0)
std_m_node = np.std(mags_node, axis=0)
std_m_link = np.std(mags_link, axis=0)

axes[0, 1].plot(times_common, mean_m_node, 'b-', lw=2, label='Dinámica de nodos')
axes[0, 1].fill_between(times_common, mean_m_node - std_m_node, mean_m_node + std_m_node,
                        color='blue', alpha=0.2)
axes[0, 1].plot(times_common, mean_m_link, 'r-', lw=2, label='Dinámica de aristas')
axes[0, 1].fill_between(times_common, mean_m_link - std_m_link, mean_m_link + std_m_link,
                        color='red', alpha=0.2)
axes[0, 1].axhline(m0_target, color='green', linestyle='--', lw=2, label=f'm(0) = {m0_target}')
axes[0, 1].set_xlabel('Paso MC')
axes[0, 1].set_ylabel(r'$\langle m \rangle$')
axes[0, 1].set_title('Magnetización promedio')
axes[0, 1].legend()

# Panel 3: Histograma de m final
m_final_node = mags_node[:, -1]
m_final_link = mags_link[:, -1]

axes[1, 0].hist(m_final_node, bins=20, alpha=0.5, color='blue', label='Dinámica de nodos', density=True)
axes[1, 0].hist(m_final_link, bins=20, alpha=0.5, color='red', label='Dinámica de aristas', density=True)
axes[1, 0].axvline(m0_target, color='green', linestyle='--', lw=2, label=f'm(0) = {m0_target}')
axes[1, 0].axvline(np.mean(m_final_node), color='blue', linestyle='-', lw=2)
axes[1, 0].axvline(np.mean(m_final_link), color='red', linestyle='-', lw=2)
axes[1, 0].set_xlabel('m final')
axes[1, 0].set_ylabel('Densidad')
axes[1, 0].set_title(f'Distribución de m final (t = {times_common[-1]:.0f})')
axes[1, 0].legend()

# Panel 4: Evolución de la varianza de m (en lugar de texto)
var_m_node = np.var(mags_node, axis=0)
var_m_link = np.var(mags_link, axis=0)

axes[1, 1].plot(times_common, var_m_node, 'b-', lw=2, label='Dinámica de nodos')
axes[1, 1].plot(times_common, var_m_link, 'r-', lw=2, label='Dinámica de aristas')
axes[1, 1].set_xlabel('Paso MC')
axes[1, 1].set_ylabel('Var(m)')
axes[1, 1].set_title('Varianza de m entre realizaciones')
axes[1, 1].legend()

plt.tight_layout()
plt.show()

print(f"\nFigura 2: Test de conservación de magnetización en red Barabási-Albert.")
print(f"Modelo: N = {N}, m = {m_BA}, m(0) = {m0_target}, {n_trials} realizaciones.")
print(f"")
print(f"Estadísticas:")
print(f"  Dinámica de NODOS:")
print(f"    <m> final = {np.mean(m_final_node):.3f}, σ(m) = {np.std(m_final_node):.3f}")
print(f"    Consenso alcanzado: {100*np.mean(np.abs(m_final_node) > 0.9):.0f}% de realizaciones")
print(f"  Dinámica de ARISTAS:")
print(f"    <m> final = {np.mean(m_final_link):.3f}, σ(m) = {np.std(m_final_link):.3f}")
print(f"    Consenso alcanzado: {100*np.mean(np.abs(m_final_link) > 0.9):.0f}% de realizaciones")
print(f"")
print(f"Conclusión: La dinámica de nodos viola la conservación de m en redes heterogéneas.")
print(f"Los hubs (nodos de alto grado) son copiados más frecuentemente, sesgando la dinámica.")

# ============================================
# Visualización de la dinámica en red pequeño mundo
# ============================================

# Crear red pequeña para visualizar
N_vis = 1000
k_vis = 4

fig, axes = plt.subplots(2, 4, figsize=(16, 8))

for row, p in enumerate([0, 0.5]):
    adj_list = create_watts_strogatz(N_vis, k_vis, p, seed=42)
    neighbors, neighbor_ptr, degrees = adj_list_to_arrays(adj_list)
    
    np.random.seed(42)
    states = np.random.choice([-1, 1], size=N_vis).astype(np.int64)
    
    # Posiciones en círculo
    theta = np.linspace(0, 2*np.pi, N_vis, endpoint=False)
    pos_x = np.cos(theta)
    pos_y = np.sin(theta)
    
    snapshot_times = [0, 50, 200, 1000]
    
    current_step = 0
    snapshot_idx = 0
    
    while snapshot_idx < len(snapshot_times):
        if current_step == snapshot_times[snapshot_idx]:
            # Visualizar
            ax = axes[row, snapshot_idx]
            
            colors = ['red' if s == -1 else 'blue' for s in states]
            ax.scatter(pos_x, pos_y, c=colors, s=30, zorder=5)
            
            # Dibujar algunas aristas
            for i in range(N_vis):
                start = neighbor_ptr[i]
                end = neighbor_ptr[i + 1]
                for idx in range(start, end):
                    j = neighbors[idx]
                    if j > i:  # Evitar duplicados
                        ax.plot([pos_x[i], pos_x[j]], [pos_y[i], pos_y[j]], 
                                'gray', alpha=0.1, lw=0.5)
            
            ax.set_xlim(-1.3, 1.3)
            ax.set_ylim(-1.3, 1.3)
            ax.set_aspect('equal')
            ax.axis('off')
            
            m = compute_magnetization(states, N_vis)
            ax.set_title(f't = {current_step}, m = {m:.2f}')
            
            if snapshot_idx == 0:
                ax.text(-1.5, 0, f'p = {p}', fontsize=12, fontweight='bold',
                        rotation=90, va='center')
            
            snapshot_idx += 1
        
        # Avanzar simulación
        if current_step < snapshot_times[-1]:
            for _ in range(N_vis):
                states = voter_step(states, neighbors, neighbor_ptr, degrees, N_vis)
            current_step += 1
        else:
            break

plt.suptitle('Evolución del modelo del votante en red pequeño mundo', fontsize=14, y=1.02)
plt.tight_layout()
plt.show()

print(f"\nFigura 3: Snapshots de la evolución del modelo del votante.")
print(f"Modelo: N = {N_vis}, k = {k_vis}. Azul = +1, Rojo = -1.")
print(f"p = 0: Los dominios crecen hasta alcanzar consenso.")
print(f"p = 0.5: Los atajos fragmentan los dominios, impidiendo consenso completo.")
