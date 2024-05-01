from dash import Dash, html, dcc, callback, Output, Input, Patch, clientside_callback
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
import pandas as pd
import dash_bootstrap_components as dbc
from dash_bootstrap_templates import load_figure_template
import numpy as np
import matplotlib.pyplot as plt
from cylinder import load_data, change_parameters, data, plot, error, coef_pd, load_model_cyl
from fraenkels import load_model_fr
plt.style.use('ggplot')

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.FONT_AWESOME])


def plot_cylinder(inputs, psi, Ux, Uy, c, U, r, model, N=250, mode='velocity',):
    fig = go.Figure()
    
    inputs = inputs.detach().numpy()
    X, Y = inputs[:,0].reshape(N,N), inputs[:,1].reshape(N,N)
    if mode == 'velocity':
        fig = px.imshow((np.sqrt(Ux**2+Uy**2)).T, aspect='equal', origin='lower',color_continuous_scale='viridis', zmin=0, zmax=2*U, x=np.linspace(-3,3,N), y=np.linspace(-3,3,N))

    elif mode == 'streamline':
        fig.add_contour(z=psi+U*Y, x=np.linspace(-3,3,N), y=np.linspace(-3,3,N), ncontours=30, opacity=1, contours = dict(showlabels = False))
    fig.add_shape(type="circle", layer='above', editable=False,
        fillcolor="black",
        x0=-r, y0=-r, x1=r, y1=r,
        xref="x", yref="y",
        opacity=0.35
        )
    err = error(X, Y, c, U, r, model)
    cl, cl_pred, cd, cd_pred  = coef_pd(c, U, r, model)
    cl_err =  np.linalg.norm(cl-cl_pred)
    cd_err =  np.linalg.norm(cd-cd_pred)
    fig.add_shape(type='rect', editable=False,
        fillcolor="white",
        xref="x", yref="y",
        x0=1, y0=2.1, x1=2.95, y1=2.85,
        opacity=0.2, 
        )

    text =[f'error (rel l2) = {err:.2f}%', 
           f"C_l error (l2) = {cl_err:.2f}", 
           f'C_d error (l2) = {cd_err:.2f}']
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

    fig.update_layout(width=800, height=800, xaxis_title = 'x', yaxis_title='y', paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",modebar = dict(bgcolor='rgba(0, 0, 0, 0)'))
    fig.update_layout(
    font=dict(
        family="Arial",
        size=16,  # Set the font size here
        # color="black"
    )
)
    return fig

def plot_fraenkels(inputs, psi, Ux, Uy, v, U, r, model, N=250, mode='velocity',):
    fig = go.Figure()
    
    inputs = inputs.detach().numpy()
    X, Y = inputs[:,0].reshape(N,N), inputs[:,1].reshape(N,N)
    if mode == 'velocity':
        fig = px.imshow((np.sqrt(Ux**2+Uy**2)).T, aspect='equal', origin='lower',color_continuous_scale='viridis', zmin=0, zmax=2*U, x=np.linspace(-3,3,N), y=np.linspace(-3,3,N))

    elif mode == 'streamline':
        fig.add_contour(z=psi+U*Y, x=np.linspace(-2.5,2.5,N), y=np.linspace(0,2.5,N), ncontours=30, opacity=1, contours = dict(showlabels = False))
    fig.add_shape(type="circle", layer='above', editable=False,
        fillcolor="black",
        x0=-r, y0=-r, x1=r, y1=r,
        xref="x", yref="y",
        opacity=0.35
        )
    err = error(X, Y, v, U, r, model)

    fig.add_shape(type='rect', editable=False,
        fillcolor="white",
        xref="x", yref="y",
        x0=1, y0=2.3, x1=2.3, y1=1.85,
        opacity=0.2, 
        )

    text =[f'error (rel l2) = {err:.2f}%']
    for i,x in enumerate([1.7]):
        fig.add_annotation(dict(font=dict(color='black',size=15),
                                            x=x,
                                            y=[2.1][i],
                                            showarrow=False,
                                            text=text[i],
                                            textangle=0,
                                            xanchor='center',
                                            xref="x",
                                            yref="y"))

    fig.update_layout(width=800, height=800, xaxis_title = 'x', yaxis_title='y', paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",modebar = dict(bgcolor='rgba(0, 0, 0, 0)'))
    fig.update_layout(
    font=dict(
        family="Arial",
        size=16,  # Set the font size here
        # color="black"
    )
)
    return fig


# fig = go.Figure()

N = 250
init_circ = 0
init_u = 1
init_radius = 1
init_data = 100
initial_layers = [6]*1
model = load_model_cyl(data_size=init_data, layers=initial_layers)
inputs, psi, Ux, Uy = data(-3, 3, -3, 3, init_circ, init_u, init_radius, model, N=N,)

fig = plot_cylinder(inputs, psi, Ux, Uy, c=0, U=1, r=1, model=model, N=250, mode='streamline')

# adds  templates to plotly.io
load_figure_template(["minty", "minty_dark"])

color_mode_switch =  html.Span(
    [
        dbc.Label(className="fa fa-moon", html_for="color-mode-switch"),
        dbc.Switch( id="color-mode-switch", value=False, className="d-inline-block ms-1", persistence=True),
        dbc.Label(className="fa fa-sun", html_for="color-mode-switch"),
    ]
)

app.layout = dbc.Container([
        html.Div([
            html.H1(children='Trained Models', style={'textAlign':'center'}),
             dcc.Dropdown(["Flat Plate", "PF Cylinder w/ circulation", "Fraenkel's Cylinder"], "PF Cylinder w/ circulation", id='dropdown-selection'),
            # dcc.Tabs(id="tabs", value='tab-2', children=[
            #     dcc.Tab(label='Flat Plate', value='tab-1'),
            #     dcc.Tab(label='PF Cylinder w/ circulation', value='tab-2'),
            #     dcc.Tab(label="Fraenkel's Cylinder", value='tab-3'),
            #         ]),
            ]),
        html.Div(children=[
            dcc.Graph(id="graph", figure= fig),
            ],
            style={'display': 'inline-block', 'vertical-align': 'top', 'margin-left': '-10vw', 'margin-top': '-3vw'}
            ),
        html.Div(children=[
            html.Label('Radius:', style={'textAlign':'center'}),
            dcc.Slider(min=.5,max=3,value=1, marks={i: f'{i}' for i in np.arange(.5,3,.5)}, id='radius-slider'),
            html.Label('Circulation:', style={'textAlign':'center'}),
            dcc.Slider(min=-5,max=5,value=0, marks={i: f'{i}' for i in np.arange(-5,5,1.)}, id='circ-slider'),
            html.Label('U:', style={'textAlign':'center'}),
            dcc.Slider(min=.5,max=5,value=1, marks={i: f'{i}' for i in np.arange(.5,5.5,.5)}, id='u-slider'),
            ],
            style={ 'margin-left': '37vw', 'margin-top': '-15vw', 'width': '60%'}),
        html.Div(children=[
            dbc.Button("Contour", outline=True, color="secondary", className="me-1", id='contour-button'),
            dbc.Button("Velocity", outline=True, color="secondary", className="me-1",  id='velocity-button'),
            dbc.Button("3D", outline=True, color="secondary", className="me-1",  id='3d-button'),
            ],
            
            style={'margin-left': '50vw', 'margin-top': '-30vw'}
),
        html.Div(children=[
            color_mode_switch
            ], 
            style={'margin-left': '-11vw', 'margin-top': '-14.5vw'}
            ),
        html.Div(children=[
            html.Label('Number of training points', style={'textAlign':'center'}),
            dcc.RadioItems(['100', '1,000', '10,000'], '100', labelStyle={'label': 'Number of points'}, inline=True, id="number-points"),
            html.Label('Size of NN', style={'textAlign':'center'}),
            dcc.RadioItems(['[6]*1', '[16]*2'], '[6]*1', inline=True, id="size-nn"),
            ], 
            style={'margin-left': '50vw', 'margin-top': '15vw',}
            ),

        ])

@callback(
    Output('graph', 'figure'),
    Input('dropdown-selection', 'value'),
    Input('radius-slider', 'value'),
    Input('circ-slider', 'value'),
    Input('u-slider', 'value'),
    Input('number-points', 'value'),
    Input('size-nn', 'value'),
)
def update_graph(testcase, radius, circ, u, n_points, size_nn):
    if testcase == "PF Cylinder w/ circulation":
        n_points = int(n_points.replace(',', ''))
        if size_nn == '[6]*1': size_nn = [6]*1
        elif size_nn == '[16]*2': size_nn = [16]*2
        N = 250 
        model = load_model_cyl(data_size=n_points, layers=size_nn)
        inputs, psi, Ux, Uy = data(-3, 3, -3, 3, circ, u, radius, model, N=N)
        fig = plot_cylinder(inputs, psi, Ux, Uy, c=circ, U=u, r=radius, model=model, N=N, mode='streamline')
        return fig
    elif testcase == "Flat Plate":
        # n_points = int(n_points.replace(',', ''))
        # if size_nn == '[6]*1': size_nn = [6]*1
        # elif size_nn == '[16]*2': size_nn = [16]*2
        # N = 250 
        # model = load_model(data_size=n_points, layers=size_nn)
        # inputs, psi, Ux, Uy = data(-3, 3, -3, 3, circ, u, radius, model, N=N)
       
        # fig = plot_cylinder(inputs, psi, Ux, Uy, c=circ, U=u, r=radius, model=model, N=N, mode='streamline')
        return fig
    elif testcase == "Fraenkel's Cylinder":
        n_points = int(n_points.replace(',', ''))
        if size_nn == '[16]*2': size_nn = [16]*2
        elif size_nn == '[32]*4': size_nn = [32]*4
        N = 250 
        model = load_model_fr(data_size=n_points, layers=size_nn)
        inputs, psi, Ux, Uy = data(-2.5, 2.5, 0, 2.5, circ, u, radius, model, N=N)
       
        fig = plot_fraenkels(inputs, psi, Ux, Uy, v=circ, U=u, r=radius, model=model, N=N, mode='streamline')
        return fig
    else:
        pass

def update_figure_template(switch_on):
    # When using Patch() to update the figure template, you must use the figure template dict
    # from plotly.io  and not just the template name
    template = pio.templates["minty"] if switch_on else pio.templates["minty_dark"]

    patched_figure = Patch()
    patched_figure["layout"]["template"] = template
    return patched_figure

clientside_callback(
    """
    (switchOn) => {
       document.documentElement.setAttribute('data-bs-theme', switchOn ? 'light' : 'dark');  
       return window.dash_clientside.no_update
    }
    """,
    Output("color-mode-switch", "id"),
    Input("color-mode-switch", "value"),
)

# # Add dropdown
# fig.update_layout(
#     updatemenus=[
#         dict(
#             type = "buttons",
#             direction = "left",
#             buttons=list([
#                 dict(
#                     args=["type", "colormap"],
#                     label="Colormap",
#                     method="update"
#                 ),
#                 dict(
#                     args=["type", "contour"],
#                     label="Contour",
#                     method="update"
#                 ),
#                 dict(
#                     args=["type", "surface"],
#                     label="3D Surface",
#                     method="update"
#                 ),
#             ]),
#             pad={"r": 10, "t": 10},
#             showactive=True,
#             x=0.,
#             xanchor="left",
#             y=1.1,
#             yanchor="top"
#         ),
#     ]
# )


if __name__ == '__main__':
    app.run(debug=True)
