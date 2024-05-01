import plotly.express as px
import plotly.graph_objects as go

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from cylinder import load_data, change_parameters, data,plot, error, coef_pd
plt.style.use('ggplot')

def plot(inputs, psi, Ux, Uy, c, U, r, N=250, mode='velocity',):
    fig = go.Figure()
    
    inputs = inputs.detach().numpy()
    X, Y = inputs[:,0].reshape(N,N), inputs[:,1].reshape(N,N)
    if mode == 'velocity':
        fig = px.imshow((np.sqrt(Ux**2+Uy**2)).T, aspect='equal', origin='lower',color_continuous_scale='viridis', zmin=0, zmax=2*U, x=np.linspace(-3,3,N), y=np.linspace(-3,3,N))
    # ct = im.contour(X,Y, psi+U*Y, levels=30,linewidths=.9, colors='firebrick')
        # Add contour lines on top
    elif mode == 'streamline':
        fig.add_contour(z=psi+U*Y, contours=dict(showlines=True), x=np.linspace(-3,3,N), y=np.linspace(-3,3,N), ncontours=30, opacity=1)

    # ax.clabel(ct, inline=True, fontsize=8, fmt='%1.1e')
    fig.add_shape(type="circle", layer='above', editable=False,
        fillcolor="black",
        x0=-r, y0=-r, x1=r, y1=r,
        xref="x", yref="y",
        opacity=0.35
        )
    # cbar = plt.colorbar(im, ax=ax, label='Velocity magnitude', aspect=30, shrink=1, format='%.1f')
    err = error(X, Y, c, U, r)
    cl, cl_pred, cd, cd_pred  = coef_pd(c, U, r)
    cl_err =  np.linalg.norm(cl-cl_pred)
    cd_err =  np.linalg.norm(cd-cd_pred)
    fig.add_shape(type='rect', editable=False,
        fillcolor="white",
        xref="x", yref="y",
        x0=1, y0=2.1, x1=2.95, y1=2.85,
        opacity=0.2, 
        )

    text =[f'$error\ (rel\ l2)\ =\ {err:.2f}\%$', f"$C_l\ error\ (l2)\ =\ {cl_err:.2f}$", f'$C_d\ error\ (l2)\ =\ {cd_err:.2f}$']
    for i,x in enumerate([2, 2, 2]):
        fig.add_annotation(dict(font=dict(color='black',size=15),
                                            x=x,
                                            y=[2.7, 2.5, 2.3][i],
                                            showarrow=False,
                                            text=text[i],
                                            textangle=0,
                                            xanchor='center',
                                            xref="x",
                                            yref="y"))

    fig.update_layout(width=800, height=800, xaxis = dict(title_text='$x$'), yaxis = dict(title_text='$y$'))
    fig.update_layout(
    font=dict(
        family="Arial",
        size=16,  # Set the font size here
        color="black"
    )
)
    return fig


N = 250
init_circ = 0
init_u = 1
init_radius = 1

inputs, psi, Ux, Uy = data(-3, 3, -3, 3, init_circ, init_u, init_radius, N=N)
# im = plot(inputs, psi, Ux, Uy, c=init_circ, U=init_u, r=init_radius, N=N)
im = plot(inputs, psi, Ux, Uy, c=init_circ, U=init_u, r=init_radius, N=N, mode='streamline')
im.show()

if __name__ == '__main__':
    pass
