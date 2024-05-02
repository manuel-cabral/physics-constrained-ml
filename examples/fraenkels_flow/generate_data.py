import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
plt.style.use('ggplot')

# set seed
np.random.seed(101)

def streamfunction(x, y, vort, U_inf, R=1):
    y = np.abs(y)
    r = np.sqrt(x**2 + y**2)/R
    x, y = x/R, y/R
    U_inf /= R
    psi = vort*y**2/2 + vort/(2*np.pi)*((1-1/r**2)*y + (x*y*(r**4-1))/(2*r**4)*np.log((r**2-2*x+1)/(r**2+2*x+1)) \
                +1/2*((1+1/r**4)*(x**2-y**2)-2)*np.arctan(2*y/(r**2-1))) + U_inf*(1-1/r**2)*y 
    psi *= R**2
    return psi

def u(x,y,vort,U_inf,R=1):
    r = np.sqrt(x**2 + y**2)/R
    x, y = x/R, y/R
    U_inf /= R
    return R**2*(vort*y + vort/(2*np.pi)*(2*y**2/r**4+(1-1/r**2) + x/2*(1-1/r**4+4*y**2/r**6)*np.log((r**2-2*x+1)/(r**2+2*x+1)) \
            + 4*x**2*y**2*(1-1/r**4)/((r**2-2*x+1)*(r**2+2*x+1))-(2*y*(x**2-y**2)/r**6+(1+1/r**4)*y)*np.arctan(2*y/(r**2-1)) \
            + ((1+1/r**4)*(x**2-y**2)-2)*(x**2-y**2-1)/((r**2-1)**2+4*y**2)) \
            + U_inf*(1-(x**2-y**2)/r**4))
def v(x,y,vort,U_inf,R=1):
    r = np.sqrt(x**2 + y**2)/R
    x, y = x/R, y/R
    U_inf /= R
    return R**2*(-vort/(2*np.pi)*(2*x*y/r**4+y/2*(1-1/r**4+4*x**2/r**6)*np.log((r**2-2*x+1)/(r**2+2*x+1)) \
            + 2*x*y*(x**2-y**2-1)*(1-1/r**4)/((r**2-2*x+1)*(r**2+2*x+1)) \
            - (2*x*(x**2-y**2)/r**6-(1+1/r**4)*x)*np.arctan(2*y/(r**2-1)) \
            - 2*x*y*((1+1/r**4)*(x**2-y**2)-2)/((r**2-1)**2+4*y**2)) \
            - U_inf*2*x*y/r**4)
def velocity(x, y, vort, U_inf, R=1):
    # U_inf *= -1
    if y<0: return -u(x,-y,vort,U_inf,R), v(x,-y,vort,U_inf,R)
    else: return u(x,y,vort,U_inf,R), v(x,y,vort,U_inf,R)

velocity = np.vectorize(velocity)

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

    u, v= velocity(*points.T)

    return np.column_stack((points, u, v))

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
    len_train = 1e3
    len_val = 5e3
    len_test = 1

    # bounds = [[-2.5,2.5],[0,2.5],[0,10],[.5,5],[.1,1.5]] # x,y,vort,U,R
    bounds = [[-2.5,2.5],[-2.5,2.5],[.1,3],[0,0],[.1,1.5]] # x,y,vort,U,R
    # bounds = [[-2.5,2.5],[0,2.5],[0,10],[.5,5],[1,1]] # x,y,vort,U,R
    # bounds = [[-2.5,2.5],[0,2.5],[0,10],[.5,5],[.1,1.5]] # x,y,vort,U,R
    # bounds = [[-2,2],[0,1.25],[0,10],[.5,3],[1,1]] # x,y,vort,U,R
    center = (0,0)
    # center = (6,6)
    save_datasets('fraenkels_1e3_5e3_1_uzero', bounds, N_train=int(len_train), N_val=int(len_val), N_test=int(len_test), center=center)

if __name__=='__main__':
    main()