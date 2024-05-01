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

def cylinder(x,y, radius=.95): 
    return x**2+y**2<radius

def create_sampling(num_points, circulation=0, U_inf=1, rho=1, d=4, radius=1, std=1):
    points = []
    
    while len(points) < num_points:
        x, y = np.random.multivariate_normal(np.zeros(2), std*np.identity(2), 1)[0]
        distance = np.sqrt(x**2 + y**2)
             
        if distance - radius >= 0:
            points.append([x, y]) 
    
    l = np.linspace(-d,d, int(np.sqrt(num_points)))
    xx_, yy_ = np.meshgrid(l,l)
    mask_ = cylinder(xx_,yy_, 1)
    outside_points = np.column_stack((xx_[~mask_], yy_[~mask_]))
    points = np.vstack((points, outside_points))
    
    coord = cart2polar(*points.T)
    vel_ = velocity(*coord, circulation, U_inf)

    mask_ = cylinder(*coord)
    vel_norm = np.linalg.norm(vel_, axis=0)
    pressure_ = np.ma.masked_where(mask_, pressure(vel_norm, rho))
    streamfn_ = np.ma.masked_where(mask_, streamfunc(*coord, circulation))

    return np.column_stack((points, vel_.T, pressure_, streamfn_))

def save_to_pickle(folder, fname, data):
    data = np.nan_to_num(data)
    file_path = os.path.join(folder, fname)
    with open(f'{file_path}.pkl', 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

def save_datasets(dataset, num_sets, len_train, len_val, len_test):
    N = num_sets*(len_train+len_val)+len_test
    assert N < dataset.shape[0], 'Need to sample more points'

    dir_path = os.path.dirname(os.path.abspath(__file__))
    folder_path = os.path.join(dir_path, 'datasets')
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    random_indices = np.random.choice(dataset.shape[0], N, replace=False)
    for i in range(1, num_sets + 1):
        start_train = (i - 1) * len_train
        end_train = i * len_train
        end_val = end_train + len_val

        data_train = dataset[random_indices[start_train:end_train]]
        save_to_pickle(folder_path, f'train_set{i}', data_train)

        data_val = dataset[random_indices[end_train:end_val]]
        save_to_pickle(folder_path, f'val_set{i}', data_val)
        
    data_test = dataset[random_indices[-len_test:]]
    save_to_pickle(folder_path, 'test_set', data_test)

def main():
    # sim parameters
    U_inf = 1
    rho = 1
    circulation = -3
    # L = -rho*U_inf*circulation
    # theta0, theta1 = stagnation_pts(circulation)

    # generate data
    num_points = 50_000  # Number of points to generate
    d = 4
    radius = 1  
    std = 1
    dataset_ = create_sampling(num_points, circulation, U_inf, rho, d=d, radius=radius, std=std)

    # set datasets size
    num_sets = 5
    len_test = 4_000
    len_train = 6_500
    len_val = 2_000

    # save datasets
    save_datasets(dataset_, num_sets, len_train, len_val, len_test)

if __name__=='__main__':
    main()