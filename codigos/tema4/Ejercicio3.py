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



hh = HodgkinHuxley()  # instancia usada por HH_Reduced


class HH_Reduced:
    """
    Modelo de Hodgkin-Huxley reducido a 2 dimensiones (V, U).
    Aproximación: m ≈ m∞(V), h ≈ h∞(U), n ≈ n∞(U)
    """
    
    def __init__(self):
        # Heredar parámetros del modelo HH completo
        self.hh = HodgkinHuxley()
    
    def f(self, V, U):
        """
        Corrientes iónicas con m=m∞(V), h=h∞(U), n=n∞(U).
        """
        m = self.hh.m_inf(V)
        h = self.hh.h_inf(U)
        n = self.hh.n_inf(U)
        return self.hh.F(V, m, h, n)
    
    def g(self, V, U):
        """
        Dinámica de la variable auxiliar U.
        Combina las dinámicas de h y n ponderadas por sus derivadas parciales.
        """
        # Valores actuales
        m = self.hh.m_inf(V)
        h_U = self.hh.h_inf(U)
        n_U = self.hh.n_inf(U)
        
        # Diferencias respecto al equilibrio
        h_V = self.hh.h_inf(V)
        n_V = self.hh.n_inf(V)
        
        # Constantes de tiempo
        tau_h = self.hh.tau_h(V)
        tau_n = self.hh.tau_n(V)
        
        # Derivadas parciales de F respecto a h y n
        dF_dh = self.hh.gNa * m**3 * (V - self.hh.VNa)
        dF_dn = 4 * self.hh.gK * n_U**3 * (V - self.hh.VK)
        
        # Numerador A
        A = dF_dh * (h_V - h_U) / tau_h + dF_dn * (n_V - n_U) / tau_n
        
        # Denominador B (derivadas de f respecto a h∞ y n∞)
        # Aproximación simplificada
        epsilon = 1e-6
        df_dh = (self.f(V, U + epsilon) - self.f(V, U - epsilon)) / (2 * epsilon)
        
        if abs(df_dh) < 1e-10:
            return 0
        
        return A / df_dh if df_dh != 0 else 0
    
    def g_simplified(self, V, U):
        """
        Versión simplificada de g(V, U) basada en ajuste lineal.
        Esta aproximación es más estable numéricamente.
        """
        # Aproximación: U tiende hacia V con una constante de tiempo efectiva
        tau_eff = 5.0  # ms (promedio de tau_h y tau_n)
        return (V - U) / tau_eff
    
    def derivatives(self, t, y, I_ext):
        """Sistema de EDOs del modelo reducido."""
        V, U = y
        
        if callable(I_ext):
            I = I_ext(t)
        else:
            I = I_ext
        
        dVdt = (-self.f(V, U) + I) / self.hh.Cm
        dUdt = self.g_simplified(V, U)
        
        return [dVdt, dUdt]
    
    def simulate(self, I_ext, T=100, dt=0.01, V0=-65):
        """Simula el modelo reducido."""
        y0 = [V0, V0]  # V = U en equilibrio
        
        t_span = (0, T)
        t_eval = np.arange(0, T, dt)
        
        sol = solve_ivp(self.derivatives, t_span, y0,
                        args=(I_ext,), t_eval=t_eval, method='RK45')
        
        return sol.t, sol.y[0], sol.y[1]

# Crear instancia
hh_red = HH_Reduced()
print("Modelo HH reducido (2D) inicializado.")

# Comparación modelo completo vs reducido
I = 10  # μA/cm²
T = 100  # ms

# Simular ambos modelos
t_full, V_full, m, h, n = hh.simulate(I, T=T)
t_red, V_red, U_red = hh_red.simulate(I, T=T)

fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

# Panel 1: Comparación de V(t)
axes[0].plot(t_full, V_full, 'b-', lw=2, label='HH completo (4D)')
axes[0].plot(t_red, V_red, 'r--', lw=2, label='HH reducido (2D)')
axes[0].set_ylabel('V (mV)')
axes[0].set_title(f'Comparación: Modelo HH completo vs reducido (I = {I} μA/cm²)')
axes[0].legend()

# Panel 2: Variable auxiliar U vs V
axes[1].plot(t_red, V_red, 'b-', lw=2, label='V (reducido)')
axes[1].plot(t_red, U_red, 'g--', lw=2, label='U (variable auxiliar)')
axes[1].set_xlabel('Tiempo (ms)')
axes[1].set_ylabel('Potencial (mV)')
axes[1].legend()

plt.tight_layout()
plt.show()

print(f"\nFigura 6: Comparación del modelo HH completo (4D) vs reducido (2D).")
print(f"Modelo: Hodgkin-Huxley. I = {I} μA/cm², T = {T} ms.")
print("El modelo reducido captura la dinámica esencial aunque con pequeñas diferencias en amplitud.")

# Espacio de fases (V, U) con isoclinas
V_range = np.linspace(-80, 50, 100)
U_range = np.linspace(-80, 50, 100)
V_grid, U_grid = np.meshgrid(V_range, U_range)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for idx, I in enumerate([0, 10]):
    ax = axes[idx]
    
    # Calcular dV/dt = 0 (isoclina nula de V)
    # f(V, U) = I
    V_nullcline_V = []
    V_nullcline_U = []
    for U in U_range:
        for V in V_range:
            if abs(-hh_red.f(V, U) + I) < 5:  # Aproximación
                V_nullcline_V.append(V)
                V_nullcline_U.append(U)
                break
    
    # Isoclina dU/dt = 0 (simplificada: V = U)
    ax.plot(V_range, V_range, 'g-', lw=2, label=r'$dU/dt = 0$ (V = U)')
    
    # Calcular f(V, V) - I = 0 para la isoclina de V
    f_VV = np.array([hh_red.f(V, V) for V in V_range])
    ax.plot(V_range, V_range, 'g-', lw=2)
    
    # Aproximación de la isoclina dV/dt = 0
    # Resolvemos f(V, U) = I para varios U
    for U_fixed in np.linspace(-80, 0, 20):
        roots = []
        for V in V_range:
            if abs(-hh_red.f(V, U_fixed) + I) < 3:
                roots.append(V)
        if roots:
            ax.scatter(roots, [U_fixed]*len(roots), c='b', s=5, alpha=0.5)
    
    # Trayectoria
    t, V_traj, U_traj = hh_red.simulate(I, T=80)
    ax.plot(V_traj, U_traj, 'r-', lw=1.5, alpha=0.8, label='Trayectoria')
    ax.scatter(V_traj[0], U_traj[0], c='green', s=100, zorder=5, marker='o', label='Inicio')
    
    ax.set_xlabel('V (mV)')
    ax.set_ylabel('U (mV)')
    ax.set_title(f'Espacio de fases (I = {I} μA/cm²)')
    ax.legend(loc='upper right')
    ax.set_xlim(-80, 50)
    ax.set_ylim(-80, 50)

plt.tight_layout()
plt.show()

print("\nFigura 7: Espacio de fases del modelo HH reducido.")
print("Izquierda: I = 0 (punto fijo estable, reposo).")
print("Derecha: I = 10 μA/cm² (ciclo límite, oscilaciones).")
