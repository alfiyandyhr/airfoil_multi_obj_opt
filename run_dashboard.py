from dash import Dash, dcc, html
import dash
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import base64

# Importing summary
df_summary = pd.read_csv('summary/summary_all.csv', index_col=0)

df_base_min_max = df_summary[:3]
df_lhs = df_summary[df_summary['Method'] == 'LHS']
df_dcgan = df_summary[df_summary['Method'] == 'DCGAN']
df_mixed = df_summary[df_summary['Method'] == 'DCGAN+GF']

names = []
for G in range(1,52):
    if G == 1:
        for S in range(1,101):
            names.append(f'G{G}S{S}')
    else:
        for S in range(1,11):
            names.append(f'G{G}S{S}')
        
# Importing airfoil coordinates
folders = ['lhs', 'dcgan', 'mixed']
coord_list = {}
for folder in folders:
    if folder == 'lhs': method = 'LHS'
    elif folder == 'dcgan': method = 'DCGAN'
    else: method = 'DCGAN+GF'
    for G in range(1,52):
        if G == 1:
            if folder != 'mixed':
                for S in range(1,101):
                    coord_list[f'{method} G{G}S{S}'] = np.genfromtxt(f'database/{folder}/G{G}/G{G}S{S}_coords.dat')
        else:
            for S in range(1,11):
                coord_list[f'{method} G{G}S{S}'] = np.genfromtxt(f'database/{folder}/G{G}/G{G}S{S}_coords.dat')

# Importing baseline
coord_list['baseline'] = np.genfromtxt('database/baseline/baseline_coords.dat')
coord_list['dv_min'] = np.genfromtxt('database/dv_min/dv_min_coords.dat')
coord_list['dv_max'] = np.genfromtxt('database/dv_max/dv_max_coords.dat')

# Instantiating Dash
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = Dash(__name__, external_stylesheets=external_stylesheets)

# Create server variable with Flask server object for use with gunicorn
server = app.server

# Main Layout
app.layout = html.Div([
    html.H4(children='Multi-Objective Airfoil Optimization',
            style={'text-align': 'center'}),
    
    html.H6(children='''by @alfiyandyhr at the Institute of Fluid Science, Tohoku University.''',
             style={'text-align': 'center'}),

    html.P(children=['Minimize CD and -CL; subject to A_FFD >= 0.9*A_FFD_base', html.Br(),
                     'Re: 7.04E6, Mach = 0.73, AoA = 2', html.Br()],
           style={'text-align': 'center'}),
    
    html.Div([
        dcc.RadioItems(
            id='method-radio-items',
            options= [{'label':'LHS', 'value':'LHS'},
                      {'label':'DCGAN', 'value':'DCGAN'},
                      {'label':'DCGAN+GF', 'value':'DCGAN+GF'}],
            value='LHS',
            inline=True, labelStyle={'display':'block'})
    ], style={'display':'flex', 'justify-content':'center'}),
    
    html.Div([
        dcc.Graph(
            id='crossfilter-indicator-scatter',
            clickData={'points': [{'text': 'LHS G5S7'}]}
        )
    ], style={'width': '60%', 'display': 'block', 'margin': 'auto'}),

    html.Div([
            html.Div([
                html.Img(id='cfd-image',
                         src='',
                         style={'width':'60%',
                                'display':'inline-block',
                                'margin-left':'60px',
                                'margin-top':'20px',
                                'margin-bottom':'20px'}),
                html.Div([
                        html.P(id='cfd-report',
                        children=[]),
                ], style={'display':'inline-block', 'width':'30%','position':'absolute',
                          'margin-left':'20px', 'margin-top':'50px'}),

            ], style={'display': 'inline-block', 'width': '48%', 'position':'absolute'}),

            html.Div([
                html.H6(children='''You can pinpoint an airfoil from the list below.''',
                        style={'text-align': 'center', 'margin-top': '20px'}),
                dcc.Dropdown(
                    id='crossfilter-airfoil-candidates',
                    options=[],
                    value=''
                ),
                dcc.Graph(id='airfoil-plot',
                          style={'margin-top': '80px'}),
            ], style={'display': 'inline-block', 'width': '48%', 'float':'right'})
    ], style={'width': '98%'}),
    
])

@app.callback(
    dash.dependencies.Output('crossfilter-airfoil-candidates', 'options'),
    [
     dash.dependencies.Input('method-radio-items', 'value')
    ])

def update_crossfilter_airfoil_candidates(method):
    if method=='LHS': df = df_lhs
    elif method=='DCGAN': df = df_dcgan
    else: df = df_mixed
        
    return [{'label': i, 'value': i} for i in df['Name']]

@app.callback(
    dash.dependencies.Output('crossfilter-indicator-scatter', 'figure'),
    [
     dash.dependencies.Input('crossfilter-airfoil-candidates', 'value'),
     dash.dependencies.Input('method-radio-items', 'value')
    ])

def update_graph(airfoil_candidate_name, method):

    fig = go.Figure()
    
    if method == 'LHS': df = df_lhs
    elif method == 'DCGAN': df = df_dcgan
    else: df = df_mixed
    
    # Non-dominated designs
    fig.add_trace(
        go.Scatter(
            x=-df['CL'][df['Is_Pareto']=='Yes'],
            y=df['CD'][df['Is_Pareto']=='Yes'],
            name='Non-dominated design',
            mode='markers',
            marker={
                'size': 10,
                'color': 'red',
                'symbol': 'circle'
            },
            hoverinfo='none'
        )
    )

    # Feasible solutions
    fig.add_trace(
        go.Scatter(
            x=-df['CL'][df['Feasibility']=='Feasible'],
            y=df['CD'][df['Feasibility']=='Feasible'],
            name='Feasible design',
            mode='markers',
            marker={
                'size': 4,
                'color': 'black',
                'symbol': 'circle'
            },
            text=df['Name'][df['Feasibility']=='Feasible'].to_numpy(),
            customdata=np.stack((df['CD'][df['Feasibility']=='Feasible'], df['CL'][df['Feasibility']=='Feasible']), axis=-1),
            hovertemplate='<b>%{text}</b><br>' +
                          'C<sub>D</sub> = %{customdata[0]:.6f}<br>' +
                          'C<sub>L</sub> = %{customdata[1]:.6f}<br>'
        )
    )

    # Infeasible solutions
    fig.add_trace(
        go.Scatter(
            x=-df['CL'][df['Feasibility']=='Infeasible'],
            y=df['CD'][df['Feasibility']=='Infeasible'],
            name='Infeasible design',
            mode='markers',
            marker={
                'size': 8,
                'color': 'black',
                'symbol': 'x'
            },
            text=df['Name'][df['Feasibility']=='Infeasible'].to_numpy(),
            customdata=np.stack((df['CD'][df['Feasibility']=='Infeasible'], df['CL'][df['Feasibility']=='Infeasible']), axis=-1),
            hovertemplate='<b>%{text}</b><br>' +
                          'C<sub>D</sub> = %{customdata[0]:.6f}<br>' +
                          'C<sub>L</sub> = %{customdata[1]:.6f}<br>'
        )
    )

    # Initial designs
    fig.add_trace(
        go.Scatter(
            x=-df[:100]['CL'],
            y=df[:100]['CD'],
            name='Initial design',
            mode='markers',
            marker={
                'size': 10,
                'color': 'magenta',
                'symbol': 'circle-open'
            },
            hoverinfo='none'
        )
    )

    # The selected airfoil
    fig.add_trace(
        go.Scatter(
            x=-df[df['Name'] == airfoil_candidate_name]['CL'].to_numpy(),
            y=df[df['Name'] == airfoil_candidate_name]['CD'].to_numpy(),
            name='Pinpointed Airfoil',
            mode='markers',
            marker={
                'size': 12,
                'opacity': 1.0,
                'color': 'green',
                'symbol': 'x'
            },
            hoverinfo='none'
        )
    )

    # Baseline RAE2822
    fig.add_trace(
        go.Scatter(
            x=[-df_base_min_max['CL'][0]],
            y=[df_base_min_max['CD'][0]],
            name='Baseline - RAE2822',
            mode='markers',
            marker={
                'size': 8,
                'color': 'blue',
                'symbol': 'circle'
            },
            text=['baseline'],
            customdata=np.stack((df_base_min_max['CD'][0], df_base_min_max['CL'][0]), axis=-1).reshape(1,-1),
            hovertemplate='<b>%{text}</b><br>' +
                          'C<sub>D</sub> = %{customdata[0]:.6f}<br>' +
                          'C<sub>L</sub> = %{customdata[1]:.6f}<br>'
        )
    )
    
    fig.update_layout(
        xaxis_title='Negative Lift Coefficient (-C<sub>L</sub>)',
        yaxis_title='Drag Coefficient (C<sub>D</sub>)',
        margin={'l': 40, 'b': 30, 't': 10, 'r': 0},
        height=450,
        hovermode='closest'
    )
    
    return fig

@app.callback(
    dash.dependencies.Output('cfd-image', 'src'),
    [dash.dependencies.Input('crossfilter-indicator-scatter', 'clickData')])
def update_cfd_image(clickData):
    airfoil_name = clickData['points'][0]['text']
    
    if airfoil_name == 'baseline':
        encoded = base64.b64encode(open(f'database/baseline/baseline.png', 'rb').read())
        src_link = f'data:image/png;base64,{encoded.decode()}'
    
    else:
        airfoil_name = airfoil_name.split(' ')

        folder = 'mixed' if airfoil_name[0] == 'DCGAN+GF' else airfoil_name[0].lower()
        G = airfoil_name[1].split('S')[0]
        encoded = base64.b64encode(open(f'database/{folder}/{G}/{airfoil_name[1]}.png', 'rb').read())
        src_link = f'data:image/png;base64,{encoded.decode()}'
    
    return src_link

@app.callback(
    dash.dependencies.Output('cfd-report', 'children'),
    [dash.dependencies.Input('crossfilter-indicator-scatter', 'clickData')])
def update_cfd_report(clickData):
    airfoil_name = clickData['points'][0]['text']
    child = [f'{airfoil_name}', html.Br(),
             f"CD = {df_summary[df_summary['Name']==airfoil_name]['CD'].to_numpy()[0]:.6f}", html.Br(),
             f"CL = {df_summary[df_summary['Name']==airfoil_name]['CL'].to_numpy()[0]:.6f}", html.Br(),
             f"A_constr = {df_summary[df_summary['Name']==airfoil_name]['A_constr'].to_numpy()[0]:.6f}", html.Br(),
             f"GF_Score= {0.4-df_summary[df_summary['Name']==airfoil_name]['GF_score'].to_numpy()[0]:.6f}", html.Br(),
             f"Constr_viol = {df_summary[df_summary['Name']==airfoil_name]['Constr_viol'].to_numpy()[0]:.6f}", html.Br(),
             f"{df_summary[df_summary['Name']==airfoil_name]['Feasibility'].to_numpy()[0]} design", html.Br(),]
    
    return child

def airfoil_plot(coord_np, title):
    return {
        'data': [dict(
            x=pd.Series(coord_np[:,0]),
            y=pd.Series(coord_np[:,1]),
            mode='lines'
        )],
        'layout': {
            'height': 225,
            'margin': {'l': 20, 'b': 30, 'r': 10, 't': 10},
            'annotations': [{
                'x': 0, 'y': 0.85, 'xanchor': 'left', 'yanchor': 'bottom',
                'xref': 'paper', 'yref': 'paper', 'showarrow': False,
                'align': 'left', 'bgcolor': 'rgba(255, 255, 255, 0.5)',
                'text': title
            }],
            'yaxis': {'type': 'linear'},
            'xaxis': {'showgrid': False},
        }
    }

@app.callback(
    dash.dependencies.Output('airfoil-plot', 'figure'),
    [dash.dependencies.Input('crossfilter-indicator-scatter', 'clickData')])
def update_airfoil_plot(clickData):
    airfoil_name = clickData['points'][0]['text']
    coord_np = coord_list[airfoil_name]
    return airfoil_plot(coord_np, airfoil_name)

if __name__ == '__main__':
    app.run_server(debug=True)