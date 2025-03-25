import dash
from dash import dcc, html
import plotly.graph_objs as go
from dash.dependencies import Input, Output
import pandas as pd
import io
from functools import reduce
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm
from scipy.ndimage import zoom
from scipy.interpolate import griddata
from PIL import Image
import cv2
from tqdm import tqdm
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import seaborn as sns
import matplotlib.colors as mcolors
import pickle
import plotly.graph_objects as gosave
import dash_bootstrap_components as dbc


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css', 'https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/css/bootstrap.min.css']


#logo = image_to_base64('../.png')

# Laden Sie Ihre Daten
lunar_temp_data_analysed = pd.read_hdf('../data/lunar_temp_data_analysed_compressed', key='data')

navbar = dbc.Navbar(
    children = [
    
        # Use row and col to control vertical alignment of logo / brand
        html.Div(html.Button([
                html.Div(style={"width": "22px", "height": "2px", "background": "black"}),
                html.Div(style={"width": "22px", "height": "2px", "background": "black", "margin": "5px 0"}),
                html.Div(style={"width": "22px", "height": "2px", "background": "black"}),
            ], id="btn_sidebar", style={"background": "none", "border": "none"}),
            #className="mr-1",
            style={"padding": "20px"},
            ),
        
        html.H1('Lunar Site Selection Algorithm | Spring Institute: Forest On The Moon Mission', style={'display': 'inline-block', 'vertical-align': 'middle', 'width': 'auto', 'height': 'auto', 'margin-right': '14px', 'color': '#1E2630', 'font-size': '20px', 'line-height': '1.5'}), #'#346EFD'
        
#        html.Div([
        #html.Img(src=f'data:image/png;base64,{image_base64}', height='60', width='auto', style={'display': 'inline-block', 'vertical-align': 'middle', 'marginLeft': 'auto'}),
        html.Img(src="https://thespringinstitute.com/wp-content/themes/WP-TheSpringInstituteV1/assets/img/astronaut.png", height='80', width='auto', style={'display': 'inline-block', 'vertical-align': 'middle', 'marginLeft': 'auto'}),
#    ], style={'width': 'auto', 'height': 'auto', 'display': 'flex', 'justify-content': 'flex-start', 'align-items': 'center'}),

],
    color="#F5F5F5",
)


app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div(
    children=[
        navbar,
        html.Div([
        html.P(id="description",
               children='This application calculates the feasibility score of potential landing site areas for the Forest On The Moon Mission. The feasibility calculation depends on the feature constraint ranges defined in the sliders at the bottom of this app. The higher the feasibility score on scale from 0-1, the better the feasibility for the Forest On The Moon Missiion. \n Being an first MVP on the journey to a holisitc lunar surface analysis application, this approach covers the data from the lunar north pole, respectively Lat: 90°N to 80°N Long: 180°W to -180°E (5-July-2009 to 17-Feb-2019)', style={'font-size': '14px'}),
        ], style={'border-left': '8px solid #346EFD', 'padding-left': '10px', 'margin-top': '40px', 'margin-bottum': '10px',  'margin-left': '30px', 'margin-right': '50px', 'font-size': '10px'}),
        html.Div(
            [
                html.Div(
                    dcc.Graph(
                        id='lunar-temp-graph',
                    ),
                    style={'width': '65%', 'margin-left': '40px', 'margin-top': '10px', 'margin-right': '10px'}
                ),
                html.Div(
                    [
                        html.Label('Minimum Temperature', style={'font-size': '18px', 'margin-top': '70px'}),
                        dcc.Slider(150, 350, 1,
                            value=242,
                            id='temp-min-slider',
                            included=True,
                            tooltip={'always_visible': True, 'placement': 'bottom'},
                            marks={i: {'label': '{}°K'.format(i), 'style': {'color': 'white'}} for i in range(150, 351, 40)},
                        ),
                        html.Label('Maximum Temperature', style={'font-size': '18px', 'margin-top': '50px'}),
                        dcc.Slider(
                            id='temp-max-slider',
                            min=150,
                            max=350,
                            value=296,
                            step=1,
                            included=True,
                            tooltip={'always_visible': True, 'placement': 'bottom'},
                            marks={i: {'label': '{}°K'.format(i), 'style': {'color': 'white'}} for i in range(150, 351, 40)},
                        ),
                        html.Label('Average Temperature', style={'font-size': '18px', 'margin-top':'50px'}),
                        dcc.Slider(
                            id='temp-avg-slider',
                            min=150,
                            max=350,
                            value=264,
                            step=1,
                            included=True,
                            tooltip={'always_visible': True, 'placement': 'bottom'},
                            marks={i: {'label': '{}°K'.format(i), 'style': {'color': 'white', 'color': 'white'}} for i in range(150, 351, 40)},
                        ),

                        html.Div(
                            dcc.Markdown('''
                            
                                The feasibilty score is a measure for lunar areas indicating their potential to serve as a lading site for the Forest On The Moon Mission.
                                The feasibiliity score is normaliized on a scale from 0-1, whereas 0 indicates non-feasibility, while 1 indicates the perfect fullfillment of all constraints.

                                Feasibility Score formula (without normalization): \n
                                $f_i = 1/3 * (tmin_i - tmin_c) - 1/3 * (tmax_i - tmax_c) - 1/3 * \sqrt{(tavg_i - tavg_c)^2}$   
                                $i, observation; c, constraint$
                                

                            ''', mathjax=True), style = {'margin-top': '50px'}),
                        html.Div(
                            style={'backgroundColor': 'blue', 'width': '100%', 'margin': 'auto auto'}
                        ),
                    ],
                    style={'width': '30%', 'margin-right': '80px', 'margin-left': 'auto'}
                )
            ],
            style={'display': 'flex', 'margin': 'auto auto'}
        ),
        dcc.Store(id='window-size'),
        html.Div(id='output-data'),

    ],
    style={
        'width': 'auto',
        'height': 'auto',
        'minHeight': '101vh',
        'overflowX': 'scroll',
        'fontFamily': 'Futura',#'Futura',
        'backgroundColor': '#1E2630',
        'color': 'white',
        'padding': '40px',
        'margin': 'auto auto',
    },
)

@app.callback(
    Output('window-size', 'data'),
    [Input('output-data', 'children')]
)
def update_window_size(children):
    return {'width': '100%', 'height': '100%'}


@app.callback(
    Output('lunar-temp-graph', 'figure'),
    [Input('temp-min-slider', 'value'),
     Input('temp-max-slider', 'value'),
     Input('temp-avg-slider', 'value'),
     Input('window-size', 'data')
]
    )

def update_figure(temp_min, temp_max, temp_avg, window_size):


        # open lunar_temp_data_collection
    with open('../results/lunar_temp_data_collection', 'rb') as file:
        lunar_temp_data_collection = pickle.load(file)

    # set parameters for analysis & visualization
    coordinate_decimals = 3

    # Perform groupby across all local times
    # group measurements by unique combinations of longitude and latitude and calculate min, max and avg for every position
    # append all dataframes for all local times
    lunar_temp_data_total = pd.concat(lunar_temp_data_collection)

    # round coordinates of measurements to derive representative data for specific regions 
    lunar_temp_data_total['        clon'] = lunar_temp_data_total['        clon'].round(decimals = coordinate_decimals)
    lunar_temp_data_total['       clat'] = lunar_temp_data_total['       clat'].round(decimals = coordinate_decimals)

    # # Select 30% of the data randomly to reduce data complexity
    # sample_fraction = 0.01
    # lunar_temp_data_total = lunar_temp_data_total.sample(frac=sample_fraction).copy()

    # perform groupby to determine min, max and average temperature brightness values 
    lunar_temp_data_analysed = lunar_temp_data_total.groupby(['        clon', '       clat']).agg({'       tbol': ['min', 'max', 'mean']}).reset_index()


    lunar_temp_data_analysed['feasibility_score'] = 1/3 * (lunar_temp_data_analysed['       tbol',  'min'] - temp_min) -  1/3 * (temp_max -lunar_temp_data_analysed['       tbol',  'max'] - temp_max) - 1/3 * (np.sqrt(((temp_avg - lunar_temp_data_analysed['       tbol',  'mean'])) ** 2))
    #1/(((temp_min - lunar_temp_data_analysed['       tbol',  'min'])** 2) * 1/3 + ((temp_max - lunar_temp_data_analysed['       tbol',  'max']) ** 2) * 1/3 + ((temp_avg - lunar_temp_data_analysed['       tbol',  'mean']) ** 2) * 1/3)

    print(lunar_temp_data_analysed)

    X = lunar_temp_data_analysed['        clon']
    print(X)

    Y = lunar_temp_data_analysed['       clat']
    print(Y)

    Z = lunar_temp_data_analysed['feasibility_score']

    print(Z)

    Z_min = Z.min()
    Z_max = Z.max()
    Z_normalized = (Z - Z_min) / (Z_max - Z_min)

    Z = Z_normalized.copy()

    hist, xedges, yedges = np.histogram2d(X, Y, bins=[1000, 1000])

    xidx = np.digitize(X, xedges)
    yidx = np.digitize(Y, yedges)

    zsum = np.zeros_like(hist)

    # Create another array to hold the count of the Z values for each bin
    zcount = np.zeros_like(hist)

    # Iterate over the Z values
    for i in range(len(Z)):
        # Ignore Z values that are out of bounds
        if xidx[i] >= zsum.shape[0] or yidx[i] >= zsum.shape[1]:
            continue
        # Add the Z value to the appropriate bin
        zsum[yidx[i], xidx[i]] += Z[i]
        zcount[yidx[i], xidx[i]] += 1

    # Compute the average Z value for each bin
    zavg = zsum / np.maximum(1, zcount)

    # Now you can create your plot
    fig = go.Figure(go.Heatmap(
        x=xedges,
        y=yedges,
        z=zavg,
        colorscale='Viridis',
        zmax=Z.max(),
        zmin=Z.min()
    ))

    fig.update_layout(
    title={
        'text': f'Feasibility-Score of Lunar Areas',
        'y':0.9,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top',
        'font': {'color': 'white'},
        'pad': {'t': 10}},  # Increase this value to push title higher
    xaxis_title="Longitude",
    yaxis_title="Latitude",
    height=int(float(window_size['height']) * 0.8),
    width=int(float(window_size['width']) * 0.8),
    plot_bgcolor='#1E2630',
    paper_bgcolor='#1E2630',
    font=dict(color='white'),
    #margin=dict(t=500)
)
    xmin = min(xedges)
    xmax = max(xedges)
    x_interval = (xmax - xmin) / 10  # Replace 10 with desired number of divisions

    ymin = min(yedges)
    ymax = max(yedges)
    y_interval = (ymax - ymin) / 10  # Replace 10 with desired number of divisions

    fig.update_xaxes(range=[xmin, xmax], dtick=x_interval)
    fig.update_yaxes(range=[ymin, ymax], dtick=y_interval)

    return fig

if __name__ == '__main__':
    app.run_server(debug=True, port = 5003)
