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



def mcculloch_pitts(inputs, weights, threshold):
    """
    Neurona de McCulloch-Pitts.
    
    Parámetros:
    -----------
    inputs : array-like
        Estados de las neuronas presinápticas (0 o 1)
    weights : array-like
        Pesos sinápticos w_ij
    threshold : float
        Umbral de disparo θ
    
    Retorna:
    --------
    int : Estado de la neurona (0 o 1)
    """
    h = np.dot(weights, inputs)  # Campo local
    return 1 if h > threshold else 0

def test_gate(gate_name, weights, threshold, n_inputs=2):
    """
    Prueba una puerta lógica para todas las combinaciones de entrada.
    """
    print(f"\n{'='*40}")
    print(f"Puerta {gate_name}")
    print(f"Pesos: {weights}, Umbral: {threshold}")
    print(f"{'='*40}")
    
    if n_inputs == 1:
        inputs_list = [[0], [1]]
        print(f"{'A':^5} | {'Salida':^6}")
    else:
        inputs_list = [[0,0], [0,1], [1,0], [1,1]]
        print(f"{'A':^5} | {'B':^5} | {'Salida':^6}")
    print("-" * 25)
    
    outputs = []
    for inp in inputs_list:
        out = mcculloch_pitts(inp, weights, threshold)
        outputs.append(out)
        if n_inputs == 1:
            print(f"{inp[0]:^5} | {out:^6}")
        else:
            print(f"{inp[0]:^5} | {inp[1]:^5} | {out:^6}")
    
    return outputs

# ============================================
# PUERTA NOT: s = NOT(A)
# ============================================
# Tabla de verdad: A=0 -> 1, A=1 -> 0
# h = w*A, queremos: h > θ cuando A=0, h ≤ θ cuando A=1
# Solución: w = -1, θ = -0.5
# A=0: h = 0 > -0.5 ✓ -> 1
# A=1: h = -1 ≤ -0.5 ✓ -> 0

w_not = [-1]
theta_not = -0.5
test_gate("NOT", w_not, theta_not, n_inputs=1)

# ============================================
# PUERTA AND: s = A AND B
# ============================================
# Tabla de verdad: solo 1 cuando A=1 Y B=1
# h = w1*A + w2*B
# Solución: w1 = w2 = 1, θ = 1.5
# (0,0): h=0, (0,1): h=1, (1,0): h=1, (1,1): h=2
# Solo h=2 > 1.5

w_and = [1, 1]
theta_and = 1.5
test_gate("AND", w_and, theta_and)

# ============================================
# PUERTA OR: s = A OR B
# ============================================
# Tabla de verdad: 1 cuando A=1 O B=1 (o ambos)
# Solución: w1 = w2 = 1, θ = 0.5
# (0,0): h=0 ≤ 0.5 -> 0
# (0,1), (1,0): h=1 > 0.5 -> 1
# (1,1): h=2 > 0.5 -> 1

w_or = [1, 1]
theta_or = 0.5
test_gate("OR", w_or, theta_or)

# ============================================
# PUERTA XOR: s = A XOR B
# ============================================
# Tabla de verdad: 1 cuando A≠B
# XOR NO es linealmente separable -> necesita 2 capas
# XOR = (A OR B) AND NOT(A AND B)
# Capa 1: neurona OR y neurona NAND
# Capa 2: neurona AND que combina las salidas

print(f"\n{'='*40}")
print("Puerta XOR (requiere 2 capas)")
print("XOR = (A OR B) AND NOT(A AND B)")
print(f"{'='*40}")

def xor_network(A, B):
    """
    Red de 2 capas para implementar XOR.
    Capa 1: OR y NAND en paralelo
    Capa 2: AND de las salidas
    """
    # Capa 1
    or_out = mcculloch_pitts([A, B], [1, 1], 0.5)      # A OR B
    nand_out = mcculloch_pitts([A, B], [-1, -1], -1.5)  # NOT(A AND B)
    
    # Capa 2
    xor_out = mcculloch_pitts([or_out, nand_out], [1, 1], 1.5)  # AND
    
    return xor_out, or_out, nand_out

print(f"{'A':^5} | {'B':^5} | {'OR':^5} | {'NAND':^5} | {'XOR':^6}")
print("-" * 40)
for A, B in [(0,0), (0,1), (1,0), (1,1)]:
    xor_out, or_out, nand_out = xor_network(A, B)
    print(f"{A:^5} | {B:^5} | {or_out:^5} | {nand_out:^5} | {xor_out:^6}")

# Visualización de las regiones de decisión
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

def plot_decision_region(ax, weights, threshold, title, gate_outputs):
    """Visualiza la región de decisión de una puerta lógica."""
    # Crear malla
    x = np.linspace(-0.5, 1.5, 100)
    y = np.linspace(-0.5, 1.5, 100)
    X, Y = np.meshgrid(x, y)
    
    # Calcular campo local
    H = weights[0]*X + weights[1]*Y
    
    # Región de decisión
    Z = (H > threshold).astype(int)
    
    # Plot
    ax.contourf(X, Y, Z, levels=[-0.5, 0.5, 1.5], colors=['lightcoral', 'lightgreen'], alpha=0.6)
    ax.contour(X, Y, H, levels=[threshold], colors='black', linewidths=2, linestyles='--')
    
    # Puntos de entrada
    inputs = [(0,0), (0,1), (1,0), (1,1)]
    for i, (a, b) in enumerate(inputs):
        color = 'green' if gate_outputs[i] == 1 else 'red'
        ax.scatter(a, b, s=200, c=color, edgecolors='black', linewidths=2, zorder=5)
        ax.annotate(f'({a},{b})→{gate_outputs[i]}', (a+0.1, b+0.1), fontsize=10)
    
    ax.set_xlim(-0.3, 1.3)
    ax.set_ylim(-0.3, 1.3)
    ax.set_xlabel('Entrada A')
    ax.set_ylabel('Entrada B')
    ax.set_title(f'{title}\n$w = {weights}$, $\\theta = {threshold}$')
    ax.set_aspect('equal')

# AND
plot_decision_region(axes[0,0], [1,1], 1.5, 'AND', [0,0,0,1])

# OR
plot_decision_region(axes[0,1], [1,1], 0.5, 'OR', [0,1,1,1])

# NAND (para XOR)
plot_decision_region(axes[1,0], [-1,-1], -1.5, 'NAND', [1,1,1,0])

# XOR (no linealmente separable)
ax = axes[1,1]
ax.set_xlim(-0.3, 1.3)
ax.set_ylim(-0.3, 1.3)
xor_outputs = [0, 1, 1, 0]
inputs = [(0,0), (0,1), (1,0), (1,1)]
for i, (a, b) in enumerate(inputs):
    color = 'green' if xor_outputs[i] == 1 else 'red'
    ax.scatter(a, b, s=200, c=color, edgecolors='black', linewidths=2, zorder=5)
    ax.annotate(f'({a},{b})→{xor_outputs[i]}', (a+0.1, b+0.1), fontsize=10)
ax.set_xlabel('Entrada A')
ax.set_ylabel('Entrada B')
ax.set_title('XOR (no linealmente separable)\nRequiere red multicapa')
ax.set_aspect('equal')
ax.text(0.5, -0.15, 'No existe línea que separe\nlos puntos verdes de los rojos', 
        ha='center', fontsize=9, style='italic')

plt.tight_layout()
plt.show()

print("\nFigura 1: Regiones de decisión para puertas lógicas McCulloch-Pitts.")
print("La línea discontinua representa el umbral h = θ. Verde = salida 1, Rojo = salida 0.")
print("XOR no es linealmente separable y requiere una red de 2 capas.")
