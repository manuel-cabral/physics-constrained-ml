import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
plt.style.use('ggplot')

# set seed
np.random.seed(101)


def velocity(r,theta, circulation=0, U_inf=1, R=1):
    u = (1-np.cos(2*theta)*R**2/r**2)*U_inf - circulation*np.sin(theta)/(2*np.pi*r)
    v = -R**2/r**2*np.sin(2*theta)*U_inf + circulation*np.cos(theta)/(2*np.pi*r)
    return np.array((u, v))

def streamfunc(r,theta, circulation=0, U_inf=1, R=1):
    return (1-R**2/r**2)*r*U_inf*np.sin(theta) - circulation/(2*np.pi)*np.log(r/R)

def pressure(u, rho=1, p_inf=0, U_inf=1):
    return .5*rho*(U_inf**2-u**2)+p_inf

def stagnation_pts(circulation, U_inf=1, R=1):
    theta = 180/np.pi*np.arcsin(circulation/(4*np.pi*U_inf*R))
    return (theta, 180-theta)

def cart2polar(x,y):
    return (np.sqrt(x**2 + y**2),np.arctan2(y, x))


def sample_points(N_points, bounds):
    points = np.zeros((N_points, len(bounds)))
    j = 0
    for i,bound in enumerate(bounds[2:]):
        points[:,i+2] = np.random.uniform(*bound,size=(N_points,))
    while j < N_points:
        x, y = np.random.uniform(*bounds[0],size=(1,)), np.random.uniform(*bounds[1],size=(1,))
        distance = np.sqrt(x**2 + y**2)
             
        if distance - points[j,4] >= 0: # points[j,4] is the radius
            points[j,0], points[j,1] = x,y
            j += 1

    coord = cart2polar(*points.T[:2])
    vel = velocity(*coord, *points.T[2:])

    return np.column_stack((points, vel.T))

def save_to_pickle(folder, fname, data):
    data = np.nan_to_num(data)
    file_path = os.path.join(folder, fname)
    with open(f'{file_path}.pkl', 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

def save_datasets(name, bounds, folder='datasets', N_train=int(5e2), N_val=int(1e3), N_test=1, center=(0,0)):
    dir_path = os.path.dirname(os.path.abspath(__file__))
    folder_path = os.path.join(dir_path, folder)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    N_points = N_train + N_val + N_test
    complete = sample_points(N_points,bounds)

    complete[:,:2] += center

    dataset = {}
    dataset['train'], dataset['val'], dataset['test'] = complete[:N_train], complete[N_train:N_train+N_val], complete[N_train+N_val:]

    save_to_pickle(folder_path, name, dataset)
    return

def load_pickle(path):
    with open(f'{path}.pkl', 'rb') as f:
        data = pickle.load(f)
    return data

def main():
    len_train = 1e4
    len_val = 5e3
    len_test = 1

    bounds = [[-3,3],[-3,3],[-5,5],[.5,5],[.1,3]] # x,y,c,U,r
    center = (0,0)
    # center = (6,6)
    save_datasets('data_1e4_5e3_1_idx0', bounds, N_train=int(len_train), N_val=int(len_val), N_test=int(len_test), center=center)

if __name__=='__main__':
    main()