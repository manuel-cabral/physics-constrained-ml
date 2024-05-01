import dash
from dash import dcc, html

app = dash.Dash()
app.layout = html.Div([
    dcc.Markdown('$Area (m^{2})$', mathjax=True),
])
app.run_server()