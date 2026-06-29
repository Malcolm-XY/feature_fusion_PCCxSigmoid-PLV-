# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 16:10:34 2026

@author: 18307
"""

import numpy as np
import matplotlib.pyplot as plt

# Parameters
k = 200
tau = 0.3

# x-axis
fn_modifier = np.linspace(-1, 1, 1000)

# Your definition
# Sigmoid
alpha = (
    1.0 / (1.0 + np.exp(-k * (fn_modifier - tau)))
    - 1.0 / (1.0 + np.exp(k * (fn_modifier + tau)))
)

# Heaviside
alpha = (
    (fn_modifier > tau).astype(float)
    + (fn_modifier > -tau).astype(float)
    - 1.0
)

# Plot
plt.figure(figsize=(7,4))
plt.plot(fn_modifier, alpha, linewidth=2, label="alpha")

plt.axhline(0, color='k', linestyle='--', linewidth=0.8)
plt.axvline(0, color='k', linestyle='--', linewidth=0.8)
plt.axvline(tau, color='r', linestyle=':', label=r'$\tau$')
plt.axvline(-tau, color='r', linestyle=':')

plt.xlabel("fn_modifier")
plt.ylabel("alpha")
plt.title(f"Difference of Two Sigmoids (k={k}, tau={tau})")
plt.grid(True)
plt.legend()
plt.show()