# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 01:07:58 2026

@author: 18307
"""

import numpy as np
import matplotlib.pyplot as plt

length = 20

# Use linspace for stability
aec_sample = np.linspace(-1, 1, length)
ps_sample  = np.linspace(0, 1, length)

def pc_aec(aec, ps, k, tau):
    alpha = 1.0 / (1.0 + np.exp(-k * (ps - tau)))
    return alpha * aec

heatmap = np.zeros((length, length))

for i, aec_ in enumerate(aec_sample):
    for j, ps_ in enumerate(ps_sample):
        heatmap[i, j] = pc_aec(aec_, ps_, 50, 0.25)

plt.figure()
plt.imshow(heatmap, aspect='equal')

plt.colorbar()
plt.title("Heatmap of pc_aec")

# ✅ Smart ticks
n_ticks = 5
tick_pos = np.linspace(0, length - 1, n_ticks)

plt.xticks(
    tick_pos,
    [f"{v:g}" for v in np.linspace(0, 1, n_ticks)]
)

plt.yticks(
    tick_pos,
    [f"{v:g}" for v in np.linspace(-1, 1, n_ticks)]
)

plt.xlabel("ps")
plt.ylabel("aec")

plt.gca().invert_yaxis()

plt.tight_layout()
plt.show()