import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from scipy.linalg import null_space

# plt.style.use('ggplot')

def load_pickle(path):
    with open(f'{path}.pkl', 'rb') as f:
        data = pickle.load(f)
    # data = torch.from_numpy(data)
    return data

def load_data(name, folder='datasets'):
    dir_path = os.path.dirname(os.path.abspath(__file__))
    folder_path = os.path.join(dir_path, folder)
    dataset = load_pickle(os.path.join(folder_path,name))
    return dataset

x,y,c,U,r,u,v = load_data('data_1e4_5e3_1_idx0')['train'].T


PLOT = False
if PLOT:
    norm = np.sqrt(u**2 + v**2)
    fig, ax = plt.subplots(figsize=(10,10))
    alpha = ((c-np.min(c))/(np.max(c)-np.min(c)))**.4
    sc = ax.scatter(x,y, c=norm, s=25*U, alpha=alpha) # color = norm, size = U_inf, alpha = circulation
    # plt.colorbar(sc)
    ax.set_aspect('equal')
    center = (0,0)
    # center = (3,3)
    circle = plt.Circle(center,1.5,color='forestgreen',alpha=0.15)
    ax.add_artist(circle)
    circle = plt.Circle(center,.1,color='firebrick',alpha=0.2)
    ax.add_artist(circle)
    ax.set_xlabel('$x$', fontsize=16)
    ax.set_ylabel('$y$', fontsize=16)
    # plt.savefig('imgs/cylinder/cyl_sampling_centered.png', dpi=256, bbox_inches='tight')
    plt.show()

def cos_sim(a,b):
    return a@b/(np.linalg.norm(a)*np.linalg.norm(b))

v1 = x/y
v2 = y/r
v3 = r*U/c

def print_func(func, arr, names=None):
    # print matrix as a table
    print(' '*10, end='')
    for i in range(arr.shape[1]):
        if names is not None:
            print(f'{names[i]:^10}', end='')
        else:
            print(f'{i:>10}', end='')
    print()
    print(' '*9, '-'*(10*arr.shape[1]))
    for i in range(arr.shape[1]):
        if names is not None:
            print(f'{names[i]:>8} |', end='')
        else:
            print(f'{i:>7}', end='')
        for j in range(arr.shape[1]):
            if j>=i:
                print(f'{func(arr.T[i],arr.T[j]):^10.2f}', end='')
            else:
                print(f'{"":<10}', end='')
        print()
    print()

print_corr = lambda arr, names=None: print_func(np.corrcoef, arr, names)
print_cos = lambda arr, names=None: print_func(cos_sim, arr, names)

def phi_base(D):
    return null_space(D)

D = [[1,1,2,1,1],[0,0,-1,-1,0]] # x,y,c,U,r
phi = phi_base(D).T

new_coords = np.zeros((len(x),len(phi)))
for i,p in enumerate(phi):
    new_coords[:,i] = x**p[0] + y**p[1] + c**p[2] + U**p[3] + r**p[4]
    # print(p)

# print_corr(np.column_stack((x,y,c,U,r)))
# print_corr(np.column_stack((a,b,c)))
# print_corr(new_coords)


print()
print_cos(np.column_stack((x,y,c,U,r)), names=['x','y','c','U','r'])
print_cos(np.column_stack((v1,v2,v3)), names=['x/y', 'y/r', 'Uy/c'])
# print_cos(new_coords, names=['pi_1', 'pi_2', 'pi_3'])

print()
A = np.array([[1,-1,0,0,0],
              [0,1,0,0,-1],
              [0,1,-1,1,0]])

# A = np.array([[1,0,0,0,-1],
#               [0,1,0,0,-1],
#               [0,0,1,-1,-1]])

print(A@A.T)
print(np.linalg.det(A@A.T-np.identity(3)))
print(np.round(phi@phi.T,2))
print(np.linalg.det(phi@phi.T-np.identity(3)))


vectors = x/y, y/r, r*U/c, u, v
labels = ['$\\frac{x}{y}$','$\\frac{y}{R}$','$\\frac{U_{\infty}R}{\Gamma}$','$u$','$v$']
# labels = ['$\\frac{x}{y}$','$\\frac{y}{R}$','$\\frac{U_{\infty}R}{\Gamma}$']

# vectors = x/r, y/r, c/(r*U), u, v
# labels = ['$\\frac{x}{R}$','$\\frac{y}{R}$','$\\frac{\Gamma}{U_{\infty}R}$','$u$','$v$']
# labels = ['$\\frac{x}{R}$','$\\frac{y}{R}$','$\\frac{\Gamma}{U_{\infty}R}$']

# vectors = x,y,c,U,r,u,v
# labels = ['$x$','$y$','$\Gamma$','$U_{\infty}$','$R$','$u$','$v$']


# vectors = x/r, y/r, x/y, x*y/r**2, (U*r)/c, (U*x)/c,  (U*y)/c, (U*x*y)/(c*r)
# labels = ['$\\frac{x}{R}$','$\\frac{y}{R}$','$\\frac{x}{y}$','$\\frac{xy}{R^2}$','$\\frac{UR}{\Gamma}$','$\\frac{Ux}{\Gamma}$','$\\frac{Uy}{\Gamma}$','$\\frac{Uxy}{\Gamma R}$']
# # labels = ['$\\frac{x}{R}$','$\\frac{y}{R}$','$\\frac{x}{y}$','$\\frac{UR}{\Gamma}$','$\\frac{Ux}{\Gamma}$','$\\frac{Uy}{\Gamma}$','$\\frac{xy}{R^2}$']
# # labels = ['$\\frac{x}{R}$','$\\frac{y}{R}$','$\\frac{x}{y}$','$\\frac{UR}{\Gamma}$','$\\frac{Ux}{\Gamma}$','$\\frac{Uy}{\Gamma}$']
# # labels = ['$x$','$y$','$\Gamma$','$U_{\infty}$','$R$','$u$','$v$']

matrix = np.zeros((len(vectors),len(vectors)))
for i in range(len(vectors)):
    for j in range(len(vectors)):
        # matrix[j,i] = cos_sim(vectors[i], vectors[j])
        matrix[j,i] = np.corrcoef(vectors[i], vectors[j])[1,0]

import seaborn as sns
def get_lower_tri_heatmap(df, output="cooc_matrix.png"):
    mask = np.zeros_like(df, dtype=bool)
    mask[np.triu_indices_from(mask)] = True

    # Want diagonal elements as well
    mask[np.diag_indices_from(mask)] = False

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    # cmap = sns.diverging_palette(220, 10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns_plot = sns.heatmap(matrix, mask=mask, cmap=cmap, vmin=-1, vmax=1, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .7})
    # change ticks to labels
    ax.set_xticklabels(labels, fontsize=20)
    ax.set_yticklabels(labels, fontsize=20, rotation=0)
    
    for j in range(len(vectors)):
        for i in range(j,-1,-1):
            c = matrix[j,i]
            ax.text(i+.5, j+.5, str(round(c,2)), fontsize=20, va='center', ha='center')

    # save to file
    fig = sns_plot.get_figure()
    fig.savefig('imgs/cylinder/heathmap_vars2.png', dpi=256, bbox_inches='tight')
    plt.show()

get_lower_tri_heatmap(matrix)