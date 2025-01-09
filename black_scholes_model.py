import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def black_scholes_call(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call_price

def delta_call(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    delta = norm.cdf(d1)
    return delta

# Paramètres
S0 = 100
K = 110
r = 0.05
sigma = 0.20
T = 1

# Calcul du prix de l'option call
call_price = black_scholes_call(S0, K, T, r, sigma)
print(f"Prix de l'option call : {call_price}")

# Visualisation de Delta
S = np.linspace(50, 150, 100)
deltas = delta_call(S, K, T, r, sigma)

plt.plot(S, deltas)
plt.xlabel('Prix de l\'actif sous-jacent')
plt.ylabel('Delta')
plt.title('Delta en fonction du prix de l\'actif sous-jacent')
plt.show()