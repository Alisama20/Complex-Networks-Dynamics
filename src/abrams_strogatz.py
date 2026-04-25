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
# Modelo de Abrams-Strogatz
# ============================================

def abrams_strogatz_dm_A(m_A, t, s, a):
    """
    Ecuación dinámica para m_A (fracción de hablantes de A).
    
    dm_A/dt = s * m_A^a * (1 - m_A) - (1-s) * (1 - m_A)^a * m_A
    """
    m_B = 1 - m_A
    
    # Evitar problemas numéricos
    m_A = np.clip(m_A, 1e-10, 1 - 1e-10)
    m_B = np.clip(m_B, 1e-10, 1 - 1e-10)
    
    # Transición B -> A
    rate_to_A = s * (m_A ** a) * m_B
    
    # Transición A -> B
    rate_to_B = (1 - s) * (m_B ** a) * m_A
    
    return rate_to_A - rate_to_B


def find_fixed_points(s, a, n_points=1000):
    """
    Encuentra los puntos fijos de la dinámica.
    
    Puntos fijos: dm_A/dt = 0
    """
    m_A_range = np.linspace(0.001, 0.999, n_points)
    dm_dt = np.array([abrams_strogatz_dm_A(m, 0, s, a) for m in m_A_range])
    
    # Encontrar cruces por cero
    fixed_points = []
    for i in range(len(dm_dt) - 1):
        if dm_dt[i] * dm_dt[i+1] < 0:  # Cambio de signo
            # Interpolación lineal
            m_fp = m_A_range[i] - dm_dt[i] * (m_A_range[i+1] - m_A_range[i]) / (dm_dt[i+1] - dm_dt[i])
            fixed_points.append(m_fp)
    
    # Añadir puntos fijos triviales si existen
    if abs(abrams_strogatz_dm_A(0.001, 0, s, a)) < 0.01:
        fixed_points.append(0)
    if abs(abrams_strogatz_dm_A(0.999, 0, s, a)) < 0.01:
        fixed_points.append(1)
    
    return sorted(set([round(fp, 4) for fp in fixed_points]))


def check_stability(m_A_fp, s, a, epsilon=1e-5):
    """
    Verifica la estabilidad de un punto fijo.
    
    Estable si d(dm_A/dt)/dm_A < 0 en el punto fijo.
    """
    # Derivada numérica
    dm_plus = abrams_strogatz_dm_A(m_A_fp + epsilon, 0, s, a)
    dm_minus = abrams_strogatz_dm_A(m_A_fp - epsilon, 0, s, a)
    
    derivative = (dm_plus - dm_minus) / (2 * epsilon)
    
    return derivative < 0  # Estable si derivada negativa

print("Funciones del modelo de Abrams-Strogatz definidas.")

# ============================================
# Diagrama de flujo para diferentes valores de s y a
# ============================================

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Casos a estudiar: (s, a)
cases = [
    (0.5, 1.0),   # Prestigio igual, lineal
    (0.5, 1.5),   # Prestigio igual, baja volatilidad
    (0.5, 0.5),   # Prestigio igual, alta volatilidad
    (0.7, 1.0),   # A más prestigioso, lineal
    (0.7, 1.5),   # A más prestigioso, baja volatilidad
    (0.3, 1.5),   # B más prestigioso, baja volatilidad
]

m_A_range = np.linspace(0, 1, 500)

for idx, (s, a) in enumerate(cases):
    row = idx // 3
    col = idx % 3
    ax = axes[row, col]
    
    # Calcular dm_A/dt
    dm_dt = np.array([abrams_strogatz_dm_A(m, 0, s, a) for m in m_A_range])
    
    # Graficar
    ax.plot(m_A_range, dm_dt, 'b-', lw=2)
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.fill_between(m_A_range, 0, dm_dt, where=(dm_dt > 0), alpha=0.3, color='green')
    ax.fill_between(m_A_range, 0, dm_dt, where=(dm_dt < 0), alpha=0.3, color='red')
    
    # Encontrar y marcar puntos fijos
    fps = find_fixed_points(s, a)
    for fp in fps:
        stable = check_stability(fp, s, a)
        if stable:
            ax.scatter(fp, 0, s=150, c='green', marker='o', zorder=5, 
                       edgecolors='black', linewidths=2)
        else:
            ax.scatter(fp, 0, s=150, c='red', marker='o', zorder=5,
                       facecolors='none', edgecolors='red', linewidths=2)
    
    # Flechas indicando dirección del flujo
    for m_start in [0.1, 0.3, 0.5, 0.7, 0.9]:
        dm = abrams_strogatz_dm_A(m_start, 0, s, a)
        if abs(dm) > 0.01:
            ax.annotate('', xy=(m_start + 0.05 * np.sign(dm), 0),
                       xytext=(m_start, 0),
                       arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    
    ax.set_xlabel(r'$m_A$ (fracción hablantes A)')
    ax.set_ylabel(r'$dm_A/dt$')
    ax.set_title(f's = {s}, a = {a}')
    ax.set_xlim(0, 1)
    
    # Anotar puntos fijos
    fp_text = ', '.join([f'{fp:.2f}' for fp in fps])
    ax.text(0.02, 0.98, f'PF: {fp_text}', transform=ax.transAxes, 
            va='top', fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat'))

plt.tight_layout()
plt.show()

print("\nFigura 4: Diagrama de flujo del modelo de Abrams-Strogatz.")
print("Verde = dm_A/dt > 0 (A crece), Rojo = dm_A/dt < 0 (A decrece).")
print("Círculos verdes = puntos fijos estables, X rojas = inestables.")
print("PF = Puntos fijos.")

# ============================================
# Evolución temporal para diferentes condiciones iniciales
# ============================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

T = 50  # Tiempo total
t_eval = np.linspace(0, T, 500)

# Caso 1: s = 0.5, a = 1.5 (coexistencia imposible, prestigio igual)
ax = axes[0, 0]
s, a = 0.5, 1.5
for m0 in [0.1, 0.3, 0.49, 0.51, 0.7, 0.9]:
    sol = odeint(abrams_strogatz_dm_A, m0, t_eval, args=(s, a))
    ax.plot(t_eval, sol[:, 0], lw=2, label=f'$m_A(0)$ = {m0}')
ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
ax.set_xlabel('Tiempo')
ax.set_ylabel(r'$m_A$')
ax.set_title(f's = {s}, a = {a} (prestigio igual, baja volatilidad)')
ax.legend(loc='center right', fontsize=8)
ax.set_ylim(0, 1)

# Caso 2: s = 0.5, a = 0.5 (alta volatilidad, coexistencia posible)
ax = axes[0, 1]
s, a = 0.5, 0.5
for m0 in [0.1, 0.3, 0.5, 0.7, 0.9]:
    sol = odeint(abrams_strogatz_dm_A, m0, t_eval, args=(s, a))
    ax.plot(t_eval, sol[:, 0], lw=2, label=f'$m_A(0)$ = {m0}')
ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
ax.set_xlabel('Tiempo')
ax.set_ylabel(r'$m_A$')
ax.set_title(f's = {s}, a = {a} (prestigio igual, alta volatilidad)')
ax.legend(loc='center right', fontsize=8)
ax.set_ylim(0, 1)

# Caso 3: s = 0.7, a = 1.5 (A más prestigioso)
ax = axes[1, 0]
s, a = 0.7, 1.5
for m0 in [0.1, 0.3, 0.5, 0.7, 0.9]:
    sol = odeint(abrams_strogatz_dm_A, m0, t_eval, args=(s, a))
    ax.plot(t_eval, sol[:, 0], lw=2, label=f'$m_A(0)$ = {m0}')
ax.set_xlabel('Tiempo')
ax.set_ylabel(r'$m_A$')
ax.set_title(f's = {s}, a = {a} (A más prestigioso)')
ax.legend(loc='center right', fontsize=8)
ax.set_ylim(0, 1)

# Caso 4: s = 0.3, a = 1.5 (B más prestigioso)
ax = axes[1, 1]
s, a = 0.3, 1.5
for m0 in [0.1, 0.3, 0.5, 0.7, 0.9]:
    sol = odeint(abrams_strogatz_dm_A, m0, t_eval, args=(s, a))
    ax.plot(t_eval, sol[:, 0], lw=2, label=f'$m_A(0)$ = {m0}')
ax.set_xlabel('Tiempo')
ax.set_ylabel(r'$m_A$')
ax.set_title(f's = {s}, a = {a} (B más prestigioso)')
ax.legend(loc='center right', fontsize=8)
ax.set_ylim(0, 1)

plt.tight_layout()
plt.show()

print("\nFigura 5: Evolución temporal de m_A para diferentes condiciones iniciales.")
print("Modelo de Abrams-Strogatz de competición de lenguajes.")

# ============================================
# Diagrama de fases (s, a)
# ============================================

s_range = np.linspace(0.01, 0.99, 100)
a_range = np.linspace(0.1, 3.0, 100)

# Clasificar cada punto según el número de puntos fijos estables
phase_diagram = np.zeros((len(a_range), len(s_range)))

for i, a in enumerate(a_range):
    for j, s in enumerate(s_range):
        fps = find_fixed_points(s, a)
        
        # Contar puntos fijos estables (excluyendo 0 y 1)
        stable_interior = 0
        for fp in fps:
            if 0.01 < fp < 0.99 and check_stability(fp, s, a):
                stable_interior += 1
        
        # Clasificar
        if stable_interior > 0:
            phase_diagram[i, j] = 2  # Coexistencia
        elif s > 0.5:
            phase_diagram[i, j] = 1  # A gana
        else:
            phase_diagram[i, j] = 0  # B gana

fig, ax = plt.subplots(figsize=(10, 8))

# Plot
cmap = plt.cm.get_cmap('RdYlGn', 3)
im = ax.imshow(phase_diagram, extent=[s_range[0], s_range[-1], a_range[0], a_range[-1]],
               origin='lower', aspect='auto', cmap=cmap, vmin=-0.5, vmax=2.5)

# Línea crítica a = 1
ax.axhline(1, color='black', linestyle='--', lw=2, label='a = 1 (lineal)')

ax.set_xlabel('Prestigio s (de A)', fontsize=12)
ax.set_ylabel('Volatilidad a', fontsize=12)
ax.set_title('Diagrama de fases del modelo de Abrams-Strogatz', fontsize=14)

# Colorbar
cbar = plt.colorbar(im, ax=ax, ticks=[0, 1, 2])
cbar.ax.set_yticklabels(['B gana', 'A gana', 'Coexistencia'])

# Anotaciones
ax.text(0.25, 2.5, 'B domina', fontsize=12, ha='center', color='white', fontweight='bold')
ax.text(0.75, 2.5, 'A domina', fontsize=12, ha='center', color='white', fontweight='bold')
ax.text(0.5, 0.5, 'Coexistencia\nposible', fontsize=10, ha='center', color='black')

ax.legend(loc='upper right')

plt.tight_layout()
plt.show()

print("\nFigura 6: Diagrama de fases del modelo de Abrams-Strogatz.")
print("Parámetros: s = prestigio del lenguaje A, a = volatilidad.")
print("Para a > 1 (baja volatilidad): Uno de los lenguajes siempre se extingue.")
print("Para a < 1 (alta volatilidad): Posible coexistencia en s = 0.5.")

# ============================================
# Análisis de estabilidad: Bifurcación en a
# ============================================

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Panel 1: Diagrama de bifurcación para s = 0.5
s = 0.5
a_range = np.linspace(0.1, 2.5, 200)

stable_fps = []
unstable_fps = []

for a in a_range:
    fps = find_fixed_points(s, a)
    for fp in fps:
        if check_stability(fp, s, a):
            stable_fps.append((a, fp))
        else:
            unstable_fps.append((a, fp))

stable_fps = np.array(stable_fps)
unstable_fps = np.array(unstable_fps)

if len(stable_fps) > 0:
    axes[0].scatter(stable_fps[:, 0], stable_fps[:, 1], c='green', s=5, label='Estable')
if len(unstable_fps) > 0:
    axes[0].scatter(unstable_fps[:, 0], unstable_fps[:, 1], c='red', s=5, label='Inestable')

axes[0].axvline(1, color='gray', linestyle='--', alpha=0.7, label='a = 1')
axes[0].set_xlabel('Volatilidad a')
axes[0].set_ylabel(r'Punto fijo $m_A^*$')
axes[0].set_title(f'Diagrama de bifurcación (s = {s})')
axes[0].legend()
axes[0].set_xlim(0, 2.5)
axes[0].set_ylim(0, 1)

# Panel 2: Diagrama de bifurcación para s = 0.6
s = 0.6

stable_fps = []
unstable_fps = []

for a in a_range:
    fps = find_fixed_points(s, a)
    for fp in fps:
        if check_stability(fp, s, a):
            stable_fps.append((a, fp))
        else:
            unstable_fps.append((a, fp))

stable_fps = np.array(stable_fps)
unstable_fps = np.array(unstable_fps)

if len(stable_fps) > 0:
    axes[1].scatter(stable_fps[:, 0], stable_fps[:, 1], c='green', s=5, label='Estable')
if len(unstable_fps) > 0:
    axes[1].scatter(unstable_fps[:, 0], unstable_fps[:, 1], c='red', s=5, label='Inestable')

axes[1].axvline(1, color='gray', linestyle='--', alpha=0.7, label='a = 1')
axes[1].set_xlabel('Volatilidad a')
axes[1].set_ylabel(r'Punto fijo $m_A^*$')
axes[1].set_title(f'Diagrama de bifurcación (s = {s})')
axes[1].legend()
axes[1].set_xlim(0, 2.5)
axes[1].set_ylim(0, 1)

plt.tight_layout()
plt.show()

print("\nFigura 7: Diagramas de bifurcación del modelo de Abrams-Strogatz.")
print("Izquierda: s = 0.5 (prestigio igual). Bifurcación pitchfork en a = 1.")
print("Derecha: s = 0.6 (A más prestigioso). El lenguaje A siempre domina para a > 1.")
