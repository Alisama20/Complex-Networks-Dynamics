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

I = 10  # μA/cm²



class FitzHughNagumo:
    """
    Modelo de FitzHugh-Nagumo.
    
    dV/dt = V - V³/3 - U + I
    dU/dt = φ(V + a - bU)
    """
    
    def __init__(self, a=0.7, b=0.8, phi=0.08):
        self.a = a
        self.b = b
        self.phi = phi
    
    def V_nullcline(self, V):
        """Isoclina dV/dt = 0: U = V - V³/3 + I"""
        return V - V**3/3
    
    def U_nullcline(self, V):
        """Isoclina dU/dt = 0: U = (V + a)/b"""
        return (V + self.a) / self.b
    
    def derivatives(self, t, y, I_ext):
        """Sistema de EDOs."""
        V, U = y
        
        if callable(I_ext):
            I = I_ext(t)
        else:
            I = I_ext
        
        dVdt = V - V**3/3 - U + I
        dUdt = self.phi * (V + self.a - self.b * U)
        
        return [dVdt, dUdt]
    
    def simulate(self, I_ext, T=200, dt=0.1, V0=-1.2, U0=-0.6):
        """Simula el modelo FN."""
        y0 = [V0, U0]
        
        t_span = (0, T)
        t_eval = np.arange(0, T, dt)
        
        sol = solve_ivp(self.derivatives, t_span, y0,
                        args=(I_ext,), t_eval=t_eval, method='RK45')
        
        return sol.t, sol.y[0], sol.y[1]
    
    def find_fixed_point(self, I):
        """Encuentra el punto fijo para una corriente dada."""
        from scipy.optimize import fsolve
        
        def equations(y):
            V, U = y
            return [V - V**3/3 - U + I,
                    self.phi * (V + self.a - self.b * U)]
        
        # Punto inicial
        V0 = -1.0
        U0 = (V0 + self.a) / self.b
        
        return fsolve(equations, [V0, U0])

# Crear instancia
fn = FitzHughNagumo()
print(f"Modelo FitzHugh-Nagumo: a={fn.a}, b={fn.b}, φ={fn.phi}")

# Diferentes regímenes dinámicos
I_values = [0, 0.3, 0.5, 0.8, 1.0]
T = 300

fig, axes = plt.subplots(len(I_values), 2, figsize=(14, 3*len(I_values)))

V_range = np.linspace(-2.5, 2.5, 200)

for i, I in enumerate(I_values):
    # Simular
    t, V, U = fn.simulate(I, T=T)
    
    # Panel izquierdo: Series temporales
    axes[i, 0].plot(t, V, 'b-', lw=1.5, label='V')
    axes[i, 0].plot(t, U, 'r--', lw=1, label='U')
    axes[i, 0].set_ylabel('V, U')
    axes[i, 0].set_title(f'I = {I}', loc='left')
    axes[i, 0].legend(loc='upper right')
    
    # Panel derecho: Espacio de fases
    # Isoclinas
    V_null = fn.V_nullcline(V_range) + I  # dV/dt = 0: U = V - V³/3 + I
    U_null = fn.U_nullcline(V_range)       # dU/dt = 0: U = (V + a)/b
    
    axes[i, 1].plot(V_range, V_null, 'b-', lw=2, label=r'$dV/dt = 0$')
    axes[i, 1].plot(V_range, U_null, 'r-', lw=2, label=r'$dU/dt = 0$')
    
    # Trayectoria (últimos ciclos para mostrar atractor)
    start_idx = len(t) // 2
    axes[i, 1].plot(V[start_idx:], U[start_idx:], 'g-', lw=1, alpha=0.7)
    axes[i, 1].scatter(V[-1], U[-1], c='green', s=50, zorder=5)
    
    # Punto fijo
    fp = fn.find_fixed_point(I)
    axes[i, 1].scatter(fp[0], fp[1], c='black', s=80, marker='x', zorder=5, label='Punto fijo')
    
    axes[i, 1].set_xlabel('V')
    axes[i, 1].set_ylabel('U')
    axes[i, 1].set_xlim(-2.5, 2.5)
    axes[i, 1].set_ylim(-1, 2)
    if i == 0:
        axes[i, 1].legend(loc='upper right', fontsize=8)

axes[-1, 0].set_xlabel('Tiempo')
plt.tight_layout()
plt.show()

print("\nFigura 8: Diferentes regímenes dinámicos del modelo de FitzHugh-Nagumo.")
print(f"Parámetros: a={fn.a}, b={fn.b}, φ={fn.phi}. T={T}.")
print("Izquierda: Series temporales. Derecha: Espacio de fases con isoclinas.")

# Diagrama de bifurcación del modelo FN
I_range = np.linspace(-0.5, 1.5, 200)
V_max = []
V_min = []

print("Calculando diagrama de bifurcación del modelo FN...")

for I in I_range:
    t, V, U = fn.simulate(I, T=500)  # Tiempo largo para alcanzar atractor
    
    # Descartar transitorio
    V_steady = V[len(V)//2:]
    
    V_max.append(np.max(V_steady))
    V_min.append(np.min(V_steady))

V_max = np.array(V_max)
V_min = np.array(V_min)

# Detectar bifurcación (donde max ≠ min)
oscillating = np.abs(V_max - V_min) > 0.1

fig, ax = plt.subplots(figsize=(10, 6))

# Rama estable (punto fijo)
ax.plot(I_range[~oscillating], V_max[~oscillating], 'b-', lw=2, label='Punto fijo estable')

# Ciclo límite (máximo y mínimo)
ax.plot(I_range[oscillating], V_max[oscillating], 'r-', lw=2, label='Ciclo límite (máx)')
ax.plot(I_range[oscillating], V_min[oscillating], 'r-', lw=2, label='Ciclo límite (mín)')

# Encontrar puntos de bifurcación aproximados
transitions = np.where(np.diff(oscillating.astype(int)) != 0)[0]
for idx in transitions:
    I_bif = I_range[idx]
    ax.axvline(I_bif, color='gray', linestyle='--', alpha=0.5)
    ax.text(I_bif, ax.get_ylim()[1]*0.9, f'I≈{I_bif:.2f}', ha='center', fontsize=9)

ax.set_xlabel('Corriente I')
ax.set_ylabel('V (máx/mín)')
ax.set_title('Diagrama de bifurcación del modelo de FitzHugh-Nagumo')
ax.legend()

plt.tight_layout()
plt.show()

print(f"\nFigura 9: Diagrama de bifurcación del modelo FN (a={fn.a}, b={fn.b}, φ={fn.phi}).")
print("Línea azul: punto fijo estable. Líneas rojas: amplitud del ciclo límite.")
print("Las líneas verticales grises marcan las bifurcaciones de Hopf.")

# Excitabilidad: respuesta a pulsos de diferente amplitud
I_base = 0  # Corriente basal (en régimen excitable)
pulse_amplitudes = [0.01, 0.1, 0.1, 1]
T = 100

fig, axes = plt.subplots(2, 2, figsize=(14, 8))
axes = axes.flatten()

for i, I_pulse in enumerate(pulse_amplitudes):
    # Pulso de corriente
    def I_ext(t):
        if 10 < t < 15:
            return I_base + I_pulse
        return I_base
    
    t, V, U = fn.simulate(I_ext, T=T)
    
    # Plot
    ax = axes[i]
    ax.plot(t, V, 'b-', lw=2, label='V')
    ax.axhline(0, color='gray', linestyle='--', alpha=0.3)
    
    # Marcar pulso
    ax.axvspan(10, 15, alpha=0.2, color='yellow', label=f'Pulso I={I_pulse}')
    
    ax.set_xlabel('Tiempo')
    ax.set_ylabel('V')
    ax.set_title(f'Amplitud del pulso: I = {I_pulse}')
    ax.legend(loc='upper right')
    ax.set_xlim(0, T)

plt.suptitle('Excitabilidad en el modelo FitzHugh-Nagumo: respuesta a pulsos', y=1.02)
plt.tight_layout()
plt.show()

print("\nFigura 10: Excitabilidad del modelo FN - respuesta all-or-none.")
print(f"Parámetros: a={fn.a}, b={fn.b}, φ={fn.phi}. Pulso de 5 unidades de tiempo.")
print("Por debajo del umbral: respuesta subumbral. Por encima: spike completo.")
