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



class TsodyksMarkram:
    """
    Modelo de sinapsis dinámica de Tsodyks-Markram.
    """
    
    def __init__(self, tau_in=3.0, tau_rec=800.0, tau_fac=530.0, U_SE=0.5, A_SE=1.0):
        """
        Parámetros:
        -----------
        tau_in : float
            Tiempo de inactivación (ms)
        tau_rec : float
            Tiempo de recuperación (ms)
        tau_fac : float
            Tiempo de facilitación (ms)
        U_SE : float
            Fracción de liberación basal
        A_SE : float
            Amplitud máxima de corriente sináptica
        """
        self.tau_in = tau_in
        self.tau_rec = tau_rec
        self.tau_fac = tau_fac
        self.U_SE = U_SE
        self.A_SE = A_SE
    
    def simulate_depression(self, spike_times, T=1000, dt=0.1):
        """
        Simula sinapsis depresora (U constante).
        """
        n_steps = int(T / dt)
        t = np.arange(0, T, dt)
        
        # Variables sinápticas
        x = np.ones(n_steps)  # Recuperado
        y = np.zeros(n_steps)  # Activo
        z = np.zeros(n_steps)  # Inactivo
        
        # Corriente sináptica
        I_syn = np.zeros(n_steps)
        
        for i in range(1, n_steps):
            # Dinámica continua
            dx = z[i-1] / self.tau_rec
            dy = -y[i-1] / self.tau_in
            dz = y[i-1] / self.tau_in - z[i-1] / self.tau_rec
            
            x[i] = x[i-1] + dt * dx
            y[i] = y[i-1] + dt * dy
            z[i] = z[i-1] + dt * dz
            
            # Check si hay spike en este paso
            current_t = i * dt
            if any(abs(current_t - ts) < dt for ts in spike_times):
                # Liberación de neurotransmisores
                delta_y = self.U_SE * x[i]
                y[i] += delta_y
                x[i] -= delta_y
            
            # Corriente sináptica proporcional a y
            I_syn[i] = self.A_SE * y[i]
        
        return t, x, y, z, I_syn
    
    def simulate_facilitation(self, spike_times, T=1000, dt=0.1):
        """
        Simula sinapsis con facilitación (U dinámico).
        """
        n_steps = int(T / dt)
        t = np.arange(0, T, dt)
        
        # Variables sinápticas
        x = np.ones(n_steps)
        y = np.zeros(n_steps)
        z = np.zeros(n_steps)
        u = np.zeros(n_steps)  # Variable de facilitación
        
        I_syn = np.zeros(n_steps)
        
        for i in range(1, n_steps):
            # Dinámica continua
            dx = z[i-1] / self.tau_rec
            dy = -y[i-1] / self.tau_in
            dz = y[i-1] / self.tau_in - z[i-1] / self.tau_rec
            du = -u[i-1] / self.tau_fac
            
            x[i] = x[i-1] + dt * dx
            y[i] = y[i-1] + dt * dy
            z[i] = z[i-1] + dt * dz
            u[i] = u[i-1] + dt * du
            
            # Check si hay spike
            current_t = i * dt
            if any(abs(current_t - ts) < dt for ts in spike_times):
                # Facilitación: aumenta u
                u[i] += self.U_SE * (1 - u[i])
                # U efectivo
                U_eff = u[i] * (1 - self.U_SE) + self.U_SE
                # Liberación
                delta_y = U_eff * x[i]
                y[i] += delta_y
                x[i] -= delta_y
            
            I_syn[i] = self.A_SE * y[i]
        
        return t, x, y, z, u, I_syn


class IntegrateAndFire:
    """
    Modelo de integración y disparo lineal.
    τ_m dV/dt = -V + R_in * I_syn
    """
    
    def __init__(self, tau_m=20.0, R_in=1.0, V_th=1.0, V_reset=0.0):
        self.tau_m = tau_m
        self.R_in = R_in
        self.V_th = V_th
        self.V_reset = V_reset
    
    def simulate(self, I_syn, dt=0.1):
        """Simula el modelo IF dado la corriente sináptica."""
        n_steps = len(I_syn)
        V = np.zeros(n_steps)
        spikes = []
        
        for i in range(1, n_steps):
            dV = (-V[i-1] + self.R_in * I_syn[i-1]) / self.tau_m
            V[i] = V[i-1] + dt * dV
            
            # Check umbral (para este ejercicio no usamos spikes)
            # Solo queremos ver el EPSP
        
        return V

# Crear instancias
synapse = TsodyksMarkram(tau_in=3.0, tau_rec=800.0, U_SE=0.5)
neuron = IntegrateAndFire(tau_m=20.0, R_in=10.0)

print("Modelos inicializados:")
print(f"  Sinapsis TM: τ_in={synapse.tau_in}ms, τ_rec={synapse.tau_rec}ms, U_SE={synapse.U_SE}")
print(f"  Neurona IF: τ_m={neuron.tau_m}ms, R_in={neuron.R_in}")

# Simular respuesta a tren de spikes a diferentes frecuencias
frequencies = [10, 20, 50, 100]  # Hz
T = 500  # ms
dt = 0.1

fig, axes = plt.subplots(len(frequencies), 3, figsize=(15, 3*len(frequencies)))

for i, freq in enumerate(frequencies):
    # Generar tren de spikes
    ISI = 1000 / freq  # Intervalo inter-spike en ms
    spike_times = np.arange(50, T-50, ISI)
    
    # Simular sinapsis depresora
    t, x, y, z, I_syn = synapse.simulate_depression(spike_times, T=T, dt=dt)
    
    # Simular neurona postsináptica
    V = neuron.simulate(I_syn, dt=dt)
    
    # Panel 1: Variables sinápticas
    axes[i, 0].plot(t, x, 'b-', label='x (recuperado)')
    axes[i, 0].plot(t, y, 'r-', label='y (activo)')
    axes[i, 0].plot(t, z, 'g-', label='z (inactivo)')
    axes[i, 0].set_ylabel('Fracción')
    axes[i, 0].set_title(f'f = {freq} Hz')
    if i == 0:
        axes[i, 0].legend(loc='upper right', fontsize=8)
    axes[i, 0].set_ylim(0, 1.1)
    
    # Marcar spikes
    for ts in spike_times:
        axes[i, 0].axvline(ts, color='gray', alpha=0.3, lw=0.5)
    
    # Panel 2: Corriente sináptica
    axes[i, 1].plot(t, I_syn, 'k-')
    axes[i, 1].set_ylabel(r'$I_{syn}$')
    
    # Panel 3: EPSP (potencial postsináptico)
    axes[i, 2].plot(t, V, 'purple')
    axes[i, 2].set_ylabel('V (EPSP)')

for j in range(3):
    axes[-1, j].set_xlabel('Tiempo (ms)')

plt.suptitle('Sinapsis DEPRESORA: respuesta a diferentes frecuencias de estimulación', y=1.02)
plt.tight_layout()
plt.show()

print("\nFigura 13: Sinapsis depresora (modelo Tsodyks-Markram) + neurona IF.")
print(f"Parámetros sinapsis: τ_in={synapse.tau_in}ms, τ_rec={synapse.tau_rec}ms, U_SE={synapse.U_SE}.")
print(f"Parámetros neurona: τ_m={neuron.tau_m}ms, R_in={neuron.R_in}.")
print("La respuesta disminuye con spikes sucesivos (depresión sináptica).")

# Comparar depresión vs facilitación con parámetros que muestren la diferencia claramente
freq = 20  # Hz
T = 500
ISI = 1000 / freq
spike_times = np.arange(50, T-50, ISI)

# Sinapsis depresora: U_SE alto, recuperación lenta
syn_dep = TsodyksMarkram(tau_in=3.0, tau_rec=800.0, tau_fac=530.0, U_SE=0.5, A_SE=1.0)
t_dep, x_dep, y_dep, z_dep, I_dep = syn_dep.simulate_depression(spike_times, T=T)

# Sinapsis facilitadora: U_SE bajo, recuperación más rápida
# Para ver facilitación, τ_rec debe ser más corto para que x no se agote tanto
syn_fac = TsodyksMarkram(tau_in=3.0, tau_rec=200.0, tau_fac=200.0, U_SE=0.05, A_SE=1.0)
t_fac, x_fac, y_fac, z_fac, u_fac, I_fac = syn_fac.simulate_facilitation(spike_times, T=T)

# EPSP
V_dep = neuron.simulate(I_dep)
V_fac = neuron.simulate(I_fac)

fig, axes = plt.subplots(2, 2, figsize=(14, 8))

# Depresión
axes[0, 0].plot(t_dep, I_dep, 'b-', lw=1)
axes[0, 0].set_title(r'Sinapsis DEPRESORA ($U_{SE}$ = 0.5, $\tau_{rec}$ = 800 ms)')
axes[0, 0].set_ylabel(r'$I_{syn}$')
for ts in spike_times[:10]:
    axes[0, 0].axvline(ts, color='gray', alpha=0.3)

axes[1, 0].plot(t_dep, V_dep, 'b-', lw=1)
axes[1, 0].set_ylabel('EPSP')
axes[1, 0].set_xlabel('Tiempo (ms)')

# Facilitación
axes[0, 1].plot(t_fac, I_fac, 'r-', lw=1)
axes[0, 1].set_title(r'Sinapsis FACILITADORA ($U_{SE}$ = 0.05, $\tau_{rec}$ = 200 ms)')
for ts in spike_times[:10]:
    axes[0, 1].axvline(ts, color='gray', alpha=0.3)

axes[1, 1].plot(t_fac, V_fac, 'r-', lw=1)
axes[1, 1].set_xlabel('Tiempo (ms)')

# Limitar vista
for ax in axes.flatten():
    ax.set_xlim(0, 350)

plt.tight_layout()
plt.show()

print("\nFigura 14: Comparación depresión vs facilitación sináptica.")
print(f"Frecuencia de estimulación: {freq} Hz.")
print("Depresión (U_SE alto): la respuesta decrece con cada spike.")
print("Facilitación (U_SE bajo, τ_rec corto): la respuesta aumenta inicialmente.")
