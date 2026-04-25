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



class HodgkinHuxley:
    """
    Modelo de Hodgkin-Huxley del axón gigante del calamar.
    
    Parámetros:
    -----------
    Cm : float
        Capacitancia de membrana (μF/cm²)
    gL, gK, gNa : float
        Conductancias máximas (mS/cm²)
    VL, VK, VNa : float
        Potenciales de inversión (mV)
    """
    
    def __init__(self):
        # Parámetros estándar
        self.Cm = 1.0      # μF/cm²
        self.gL = 0.3      # mS/cm²
        self.gK = 36.0     # mS/cm²
        self.gNa = 120.0   # mS/cm²
        self.VL = -54.402  # mV
        self.VK = -77.0    # mV
        self.VNa = 50.0    # mV
    
    # Funciones de tasa α y β para cada variable de compuerta
    # (versiones numéricamente estables)
    
    def alpha_m(self, V):
        x = V + 40
        if abs(x) < 1e-6:
            return 1.0  # Límite por L'Hôpital
        exp_term = np.exp(np.clip(-0.1 * x, -700, 700))
        return 0.1 * x / (1 - exp_term)
    
    def beta_m(self, V):
        return 4.0 * np.exp(np.clip(-0.0556 * (V + 65), -700, 700))
    
    def alpha_h(self, V):
        return 0.07 * np.exp(np.clip(-0.05 * (V + 65), -700, 700))
    
    def beta_h(self, V):
        return 1.0 / (1 + np.exp(np.clip(-0.1 * (V + 35), -700, 700)))
    
    def alpha_n(self, V):
        x = V + 55
        if abs(x) < 1e-6:
            return 0.1  # Límite por L'Hôpital
        exp_term = np.exp(np.clip(-0.1 * x, -700, 700))
        return 0.01 * x / (1 - exp_term)
    
    def beta_n(self, V):
        return 0.125 * np.exp(np.clip(-0.0125 * (V + 65), -700, 700))
    
    # Valores estacionarios y constantes de tiempo
    # Valores estacionarios y constantes de tiempo
    def x_inf(self, alpha, beta):
        denom = alpha + beta
        if denom < 1e-10:
            return 0.5
        return alpha / denom
    
    def tau_x(self, alpha, beta):
        denom = alpha + beta
        if denom < 1e-10:
            return 1e10  # Valor grande pero finito
        return 1.0 / denom
    
    def m_inf(self, V):
        return self.x_inf(self.alpha_m(V), self.beta_m(V))
    
    def h_inf(self, V):
        return self.x_inf(self.alpha_h(V), self.beta_h(V))
    
    def n_inf(self, V):
        return self.x_inf(self.alpha_n(V), self.beta_n(V))
    
    def tau_m(self, V):
        return self.tau_x(self.alpha_m(V), self.beta_m(V))
    
    def tau_h(self, V):
        return self.tau_x(self.alpha_h(V), self.beta_h(V))
    
    def tau_n(self, V):
        return self.tau_x(self.alpha_n(V), self.beta_n(V))
    
    def F(self, V, m, h, n):
        """Corrientes iónicas totales."""
        IL = self.gL * (V - self.VL)
        IK = self.gK * n**4 * (V - self.VK)
        INa = self.gNa * m**3 * h * (V - self.VNa)
        return IL + IK + INa
    
    def derivatives(self, t, y, I_ext):
        """
        Sistema de EDOs del modelo HH.
        y = [V, m, h, n]
        """
        V, m, h, n = y
        
        # Limitar V a un rango razonable para evitar overflow
        V = np.clip(V, -150, 150)
        
        # Limitar variables de compuerta a [0, 1]
        m = np.clip(m, 0, 1)
        h = np.clip(h, 0, 1)
        n = np.clip(n, 0, 1)
        
        # Corriente externa (puede ser función del tiempo)
        if callable(I_ext):
            I = I_ext(t)
        else:
            I = I_ext
        
        # Derivadas
        dVdt = (-self.F(V, m, h, n) + I) / self.Cm
        
        tau_m = max(self.tau_m(V), 1e-6)
        tau_h = max(self.tau_h(V), 1e-6)
        tau_n = max(self.tau_n(V), 1e-6)
        
        dmdt = (self.m_inf(V) - m) / tau_m
        dhdt = (self.h_inf(V) - h) / tau_h
        dndt = (self.n_inf(V) - n) / tau_n
        
        return [dVdt, dmdt, dhdt, dndt]
    
    def simulate(self, I_ext, T=100, dt=0.01, V0=-65):
        """
        Simula el modelo HH.
        
        Parámetros:
        -----------
        I_ext : float o callable
            Corriente externa (μA/cm²)
        T : float
            Tiempo total de simulación (ms)
        dt : float
            Paso temporal (ms)
        V0 : float
            Potencial inicial (mV)
        
        Retorna:
        --------
        t, V, m, h, n : arrays
        """
        # Condiciones iniciales en equilibrio
        m0 = self.m_inf(V0)
        h0 = self.h_inf(V0)
        n0 = self.n_inf(V0)
        y0 = [V0, m0, h0, n0]
        
        # Integración
        t_span = (0, T)
        t_eval = np.arange(0, T, dt)
        
        sol = solve_ivp(self.derivatives, t_span, y0, 
                        args=(I_ext,), t_eval=t_eval, method='RK45')
        
        return sol.t, sol.y[0], sol.y[1], sol.y[2], sol.y[3]

# Crear instancia del modelo
hh = HodgkinHuxley()

print("Modelo de Hodgkin-Huxley inicializado.")
print(f"Parámetros: gL={hh.gL}, gK={hh.gK}, gNa={hh.gNa} mS/cm²")

# Visualizar las funciones de compuerta
V_range = np.linspace(-100, 50, 200)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Panel izquierdo: valores estacionarios
ax1 = axes[0]
ax1.plot(V_range, [hh.m_inf(V) for V in V_range], 'r-', lw=2, label=r'$m_\infty$ (act. Na⁺)')
ax1.plot(V_range, [hh.h_inf(V) for V in V_range], 'b-', lw=2, label=r'$h_\infty$ (inact. Na⁺)')
ax1.plot(V_range, [hh.n_inf(V) for V in V_range], 'g-', lw=2, label=r'$n_\infty$ (act. K⁺)')
ax1.axvline(-65, color='gray', linestyle='--', alpha=0.5, label='Reposo (-65 mV)')
ax1.set_xlabel('Potencial V (mV)')
ax1.set_ylabel('Valor estacionario')
ax1.set_title('Valores estacionarios de las compuertas')
ax1.legend()
ax1.set_xlim(-100, 50)
ax1.set_ylim(0, 1)

# Panel derecho: constantes de tiempo
ax2 = axes[1]
ax2.plot(V_range, [hh.tau_m(V) for V in V_range], 'r-', lw=2, label=r'$\tau_m$')
ax2.plot(V_range, [hh.tau_h(V) for V in V_range], 'b-', lw=2, label=r'$\tau_h$')
ax2.plot(V_range, [hh.tau_n(V) for V in V_range], 'g-', lw=2, label=r'$\tau_n$')
ax2.axvline(-65, color='gray', linestyle='--', alpha=0.5)
ax2.set_xlabel('Potencial V (mV)')
ax2.set_ylabel('Constante de tiempo (ms)')
ax2.set_title('Constantes de tiempo de las compuertas')
ax2.legend()
ax2.set_xlim(-100, 50)
ax2.set_yscale('log')

plt.tight_layout()
plt.show()

print("\nFigura 2: Propiedades de las variables de compuerta en el modelo de Hodgkin-Huxley.")
print("Izquierda: Valores estacionarios m∞, h∞, n∞ vs V.")
print("Derecha: Constantes de tiempo τm ≪ τh, τn (m es mucho más rápida).")

# Simulación para diferentes valores de corriente
I_values = [0, 5, 7, 10, 15, 20]
T = 100  # ms

fig, axes = plt.subplots(len(I_values), 1, figsize=(14, 12), sharex=True)

for i, I in enumerate(I_values):
    t, V, m, h, n = hh.simulate(I, T=T)
    
    axes[i].plot(t, V, 'k-', lw=1)
    axes[i].set_ylabel('V (mV)')
    axes[i].set_title(f'I = {I} μA/cm²', loc='left', fontsize=10)
    axes[i].set_ylim(-90, 60)
    
    # Contar spikes
    peaks, _ = find_peaks(V, height=0)
    n_spikes = len(peaks)
    axes[i].text(0.98, 0.85, f'{n_spikes} spikes', transform=axes[i].transAxes, 
                 ha='right', fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat'))

axes[-1].set_xlabel('Tiempo (ms)')
plt.tight_layout()
plt.show()

print("\nFigura 3: Respuesta del modelo HH a diferentes corrientes inyectadas.")
print(f"Modelo: Hodgkin-Huxley estándar. T = {T} ms, V₀ = -65 mV.")

# Diagrama de bifurcación: frecuencia de disparo vs corriente
I_range = np.linspace(0, 25, 100)
frequencies = []
T_sim = 500  # ms (más largo para medir frecuencia)

print("Calculando diagrama de bifurcación...")
for I in I_range:
    t, V, _, _, _ = hh.simulate(I, T=T_sim)
    
    # Detectar picos (spikes) después de transitorio
    t_transient = 100  # ms
    mask = t > t_transient
    peaks, _ = find_peaks(V[mask], height=0)
    
    if len(peaks) >= 2:
        # Calcular frecuencia como inverso del periodo medio
        t_peaks = t[mask][peaks]
        periods = np.diff(t_peaks)
        freq = 1000 / np.mean(periods)  # Hz
    else:
        freq = 0
    
    frequencies.append(freq)

frequencies = np.array(frequencies)

# Encontrar umbrales
I_threshold_low = I_range[np.where(frequencies > 0)[0][0]] if np.any(frequencies > 0) else None
I_threshold_high = I_range[np.where(frequencies > 0)[0][-1]] if np.any(frequencies > 0) else None

# Plot
fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(I_range, frequencies, 'b-', lw=2)
ax.fill_between(I_range, 0, frequencies, alpha=0.3)

if I_threshold_low:
    ax.axvline(I_threshold_low, color='r', linestyle='--', 
               label=f'Umbral inferior ≈ {I_threshold_low:.1f} μA/cm²')

ax.set_xlabel('Corriente inyectada I (μA/cm²)')
ax.set_ylabel('Frecuencia de disparo (Hz)')
ax.set_title('Diagrama de bifurcación del modelo de Hodgkin-Huxley')
ax.legend()
ax.set_xlim(0, 25)
ax.set_ylim(0, max(frequencies)*1.1)

plt.tight_layout()
plt.show()

print(f"\nFigura 4: Diagrama de bifurcación f(I) del modelo de Hodgkin-Huxley.")
print(f"Modelo: HH estándar. T_simulación = {T_sim} ms, transitorio descartado = 100 ms.")
print(f"\nRango de oscilaciones: I ∈ [{I_threshold_low:.1f}, {I_range[-1]:.1f}] μA/cm²")
print(f"Frecuencia máxima: {max(frequencies):.1f} Hz")

# Detalle de un potencial de acción
I = 10  # μA/cm²
t, V, m, h, n = hh.simulate(I, T=50)

fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

# Panel 1: Potencial de membrana
axes[0].plot(t, V, 'k-', lw=2)
axes[0].set_ylabel('V (mV)')
axes[0].set_title(f'Potencial de acción - Modelo de Hodgkin-Huxley (I = {I} μA/cm²)')
axes[0].axhline(-65, color='gray', linestyle='--', alpha=0.5, label='Reposo')
axes[0].legend(loc='upper right')

# Panel 2: Variables de compuerta
axes[1].plot(t, m, 'r-', lw=2, label='m (act. Na⁺)')
axes[1].plot(t, h, 'b-', lw=2, label='h (inact. Na⁺)')
axes[1].plot(t, n, 'g-', lw=2, label='n (act. K⁺)')
axes[1].set_ylabel('Variable de compuerta')
axes[1].legend(loc='upper right')
axes[1].set_ylim(0, 1)

# Panel 3: Corrientes iónicas
INa = hh.gNa * m**3 * h * (V - hh.VNa)
IK = hh.gK * n**4 * (V - hh.VK)
IL = hh.gL * (V - hh.VL)

axes[2].plot(t, INa, 'r-', lw=2, label=r'$I_{Na}$')
axes[2].plot(t, IK, 'g-', lw=2, label=r'$I_K$')
axes[2].plot(t, IL, 'gray', lw=1, label=r'$I_L$')
axes[2].axhline(0, color='k', linestyle='-', alpha=0.3)
axes[2].set_xlabel('Tiempo (ms)')
axes[2].set_ylabel('Corriente (μA/cm²)')
axes[2].legend(loc='upper right')

plt.tight_layout()
plt.show()

print("\nFigura 5: Anatomía de un potencial de acción en el modelo HH.")
print(f"Modelo: Hodgkin-Huxley estándar. I = {I} μA/cm², T = 50 ms.")
print("Arriba: V(t). Centro: variables de compuerta. Abajo: corrientes iónicas.")
