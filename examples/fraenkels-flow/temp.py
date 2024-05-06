import numpy as np
import matplotlib.pyplot as plt


fig, ax = plt.subplots()
d = 2.5
scale = 2.5
U_inf = 1
y = np.linspace(-d,d,25)
x = np.full_like(y, -d) 
# u = -U[::2,0]
# v = -V[::2,0]
u = U_inf*np.abs(y)+1
v = np.zeros_like(u)
ax.quiver(x,y,u,v, color='firebrick', scale=scale, width=3e-3, scale_units='inches', alpha=.7, zorder=10)
# no axis
ax.axis('off')
# save with transparent background
plt.savefig('imgs/fraenkel/velocity_profile_2.png', dpi=256, bbox_inches='tight', transparent=True)
plt.show()
