# import dash
# from dash import dcc
# from dash import html
# import plotly.graph_objs as go
# from dash.dependencies import Input, Output
# import pandas as pd
# import io
# from functools import reduce
# import requests
# import numpy as np
# import pandas as pd
# import geopandas as gpd
# import matplotlib.pyplot as plt
# import matplotlib.cm
# from scipy.ndimage import zoom
# from scipy.interpolate import griddata
# from PIL import Image
# import cv2
# from tqdm import tqdm
# from matplotlib.cm import ScalarMappable
# from matplotlib.colors import Normalize
# import seaborn as sns
# import matplotlib.colors as mcolors
# import pickle
# from dash.dependencies import Input, Output
# import plotly.graph_objects as go
# import pickle
# import pandas as pd
# import dash_bootstrap_components as dbc
# import numpy as np
# import pandas as pd
# import plotly.graph_objects as go
# import plotly.subplots as sp
# from plotly.tools import FigureFactory as FF
# from scipy.interpolate import griddata


# external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css', 'https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/css/bootstrap.min.css']


# #logo = image_to_base64('../.png')

# # Laden Sie Ihre Daten
# lunar_temp_data_analysed = pd.read_hdf('../data/lunar_temp_data_analysed_compressed', key='data')

# navbar = dbc.Navbar(
#     children = [
    
#         # Use row and col to control vertical alignment of logo / brand
#         html.Div(html.Button([
#                 html.Div(style={"width": "22px", "height": "2px", "background": "black"}),
#                 html.Div(style={"width": "22px", "height": "2px", "background": "black", "margin": "5px 0"}),
#                 html.Div(style={"width": "22px", "height": "2px", "background": "black"}),
#             ], id="btn_sidebar", style={"background": "none", "border": "none"}),
#             #className="mr-1",
#             style={"padding": "20px"},
#             ),
        
#         html.H1('Lunar Site Selection Algorithm | Spring Institute: Forest On The Moon Mission', style={'display': 'inline-block', 'vertical-align': 'middle', 'width': 'auto', 'height': 'auto', 'margin-right': '14px', 'color': '#1E2630', 'font-size': '20px', 'line-height': '1.5'}), #'#346EFD'
        
# #        html.Div([
#         #html.Img(src=f'data:image/png;base64,{image_base64}', height='60', width='auto', style={'display': 'inline-block', 'vertical-align': 'middle', 'marginLeft': 'auto'}),
#         html.Img(src="https://thespringinstitute.com/wp-content/themes/WP-TheSpringInstituteV1/assets/img/astronaut.png", height='50', width='auto', style={'display': 'inline-block', 'vertical-align': 'middle', 'marginLeft': 'auto'}),
# #    ], style={'width': 'auto', 'height': 'auto', 'display': 'flex', 'justify-content': 'flex-start', 'align-items': 'center'}),

# ],
#     color="#F5F5F5",
#     # style={"background-color": "#242D3D", "padding": "20px"},
#     # dark=True
    
# )


# app = dash.Dash(__name__, external_stylesheets=external_stylesheets)


# app.layout = html.Div(
#     children=[
#         navbar,
#         html.Div(
#             dcc.Graph(
#                 id='lunar-temp-graph',
#             ),
#             style={'display': 'flex', 'justify-content': 'center', 'margin-top': '0px'}
#         ),

#         html.Div(
#             [
#                 html.Label('Minimum Temperature', style={'font-size': '18px'}),
#                 # dcc.Slider(
#                 #     id='temp-min-slider',
#                 #     min=200,
#                 #     max=300,
#                 #     value=243,
#                 #     step=1,
#                 #     marks={i: {'label': '{}°K'.format(i), 'style': {'color': 'white'}} for i in range(200, 301, 10)},
#                 # ),
#                 dcc.Slider(150, 350, 1,
#                     value=243,
#                     id='temp-min-slider',
#                     included=True,
#                     tooltip={'always_visible': True, 'placement': 'bottom'},
#                     #handleLabel={"showCurrentValue": True,"label": "VALUE"},
#                     marks={i: {'label': '{}°K'.format(i), 'style': {'color': 'white'}} for i in range(150, 351, 20)},
#                 ),
#                 html.Label('Maximum Temperature', style={'font-size': '18px'}),
#                 dcc.Slider(
#                     id='temp-max-slider',
#                     min=150,
#                     max=350,
#                     value=303,
#                     step=1,
#                     marks={i: {'label': '{}°K'.format(i), 'style': {'color': 'white'}} for i in range(150, 351, 20)},
#                 ),
#                 html.Label('Average Temperature', style={'font-size': '18px'}),
#                 dcc.Slider(
#                     id='temp-avg-slider',
#                     min=150,
#                     max=350,
#                     value=283,
#                     step=1,
#                     marks={i: {'label': '{}°K'.format(i), 'style': {'color': 'white', 'color': 'white'}} for i in range(150, 351, 20)},
#                 ),
#                 html.Div(

#     style={'backgroundColor': 'blue', 'width': '80%', 'margin': 'auto'}
# ),

#             ],
#             style={'margin-left': '300px', 'margin-right': '300px', 'color': 'white'}
#         ),
#     ],
#     style={
#         'width': 'auto',
#         'height': 'auto',
#         'minHeight': '101vh',
#         'overflowX': 'scroll',
#         'fontFamily': 'Futura',#'Futura',
#         'backgroundColor': '#1E2630',
#         'color': 'white',
#         'padding': '40px',
#         'margin': '-8px',
#     },
# )


# @app.callback(
#     Output('lunar-temp-graph', 'figure'),
#     [Input('temp-min-slider', 'value'),
#      Input('temp-max-slider', 'value'),
#      Input('temp-avg-slider', 'value')]
# )
# def update_figure(temp_min, temp_max, temp_avg):


#         # open lunar_temp_data_collection
#     with open('../results/lunar_temp_data_collection', 'rb') as file:
#         lunar_temp_data_collection = pickle.load(file)

#     lunar_temp_data_analysed = pd.read_hdf('../data/lunar_temp_data_analysed_compressed', key='data')

#     #Create Visualization with topographic map of the moon surface

#     #fig, axs = plt.subplots(1, 3, figsize=(80, 20))
#     fig = sp.make_subplots(rows=1, cols=len(['min', 'max', 'mean']), shared_yaxes=True, subplot_titles=['Min', 'Max', 'Mean'])


#     # Read the lunar topography image using Pillow
#     lunar_north_pole_img_original = Image.open('../data/6304h.tiff')

#     #convert lunar image into numpy array and get it into the right color format for further processing the image with libaries such as OpenCV
#     lunar_img_np = np.array(lunar_north_pole_img_original)
#     lunar_north_pole_img = cv2.cvtColor(lunar_img_np, cv2.COLOR_RGB2BGR)

#     #convert lunar north pole image into greyscale format to have a baseline map for choropleth map
#     lunar_north_pole_img_grey = cv2.cvtColor(lunar_north_pole_img, cv2.COLOR_RGB2GRAY)

#     #splitting the lunar_north_pole_image binarily into tha part with the black background and the coloured part
#     #all pixel colour values under 1 (-> near black values) get the value 0, the others get the value 255
#     #based on the number assignements all pixels with the value 255 are selected to create a mask, which is the part of the picture containing the satellite map, excluding the near-black background
#     #_ stores the threshold value and is not required anymore
#     #mask stores the relevant pixel positions 
#     _, mask = cv2.threshold(lunar_north_pole_img_grey, 1, 255, cv2.THRESH_BINARY)

#     #Find the center and radius of the North Pole region
#     #1. Get identified contours of the picture (outer lines of identied forms)
#     contours, _ = cv2.findContours(mask, 
#                                 cv2.RETR_EXTERNAL, #Taking only the extreme outer contours, because contour hierarchies are not relevant in this case
#                                 cv2.CHAIN_APPROX_SIMPLE #Removes redundant point and compreses data complexity
#                                 )

#     #2. Determine the maximum area enclosed by the contour using the max-function
#     contour_area = max( #determines the maximum elememnt in an iterable 
#                     contours, #iterable = list of contours
#                     key=cv2.contourArea #key to choose largest elements: cv2.contourArea which determines the area size enclosed by a contour
#                     )

#     #3. Determine the minimum circle that encloses the contour_area and get center + radius
#     (x, y), radius = cv2.minEnclosingCircle(contour_area)

#     #performing the choropleth map creation for every operation (min, max, avg)
#     for i, operation in enumerate(['min', 'max', 'mean']):

#         lunar_temp_data_analysed_iteration = lunar_temp_data_analysed[[('        clon',     ''),( '       clat',     ''),( '       tbol', operation)]].copy()

#         #Normalization: The data must be normalized on a scale of 0-1 so that it can later be used to create a colormap
#         #1. Normalization Approach: min-max-normalization
#         #normalized_temps = (lunar_temp_data_analysed_iteration['       tbol',  operation] - np.min(lunar_temp_data_analysed_iteration['       tbol',  operation])) / (np.max(lunar_temp_data_analysed_iteration['       tbol',  operation]) - np.min(lunar_temp_data_analysed_iteration['       tbol',  operation]))

#         #2. Normalization Approach: percentile normalization (Better choice because more resistent to outliers)
#         p1 = np.percentile(lunar_temp_data_analysed_iteration['       tbol',  operation], 1)
#         p99 = np.percentile(lunar_temp_data_analysed_iteration['       tbol',  operation], 99)
#         normalized_temps = (lunar_temp_data_analysed_iteration['       tbol',  operation] - p1) / (p99 - p1)

#         lunar_temp_data_analysed_iteration['       tbol', operation] = normalized_temps.copy()    

#         #Bring data into a coordinate-grid
#         temperature_grid = lunar_temp_data_analysed_iteration.pivot(index='       clat', columns='        clon', values=('       tbol', operation))
#         temperature_grid = temperature_grid.fillna(normalized_temps.mean()).copy()

#         #Polar coordinates (clats, clons) of the temperature data
#         clats, clons = np.mgrid[80:90:(temperature_grid.shape[0] * 1j), -180:180:(temperature_grid.shape[1] * 1j)]

#         #Convert the polar coordinates to Cartesian coordinates 
#         radius_scaled = radius * (clats - 80) / 10 #calculated the scaled_radius by dividing the actual clat delta by the max clat delta
#         cart_x = radius_scaled * np.cos(np.radians(clons)) + x #radians(clons) converts degrees to radians, by multyplining the radius_scaled by the cos of that radiant the x values can be determined 
#                                                             #Imagine triangles in Cartesian coordinate system (one edge along the x-axis, one between the center and the pixel,...)
#                                                             #The + x helps to set the centre to the centre of the lunar image 
#         cart_y = radius_scaled * np.sin(np.radians(clons)) + y # "

#         coords = np.array([cart_x.ravel(), cart_y.ravel()]).T #coordinates are each flattend into 1D arrays, mapped by their index and transposed to get in the right format

#         #Data Interpolation to create a dataframe of the same size as the lunar image that contains the brightness values
#         temperature_data_mapped = griddata( #griddata performs interpolation - estimating a value based on a set of known data points to, in this case, reduce the amount of data points
#                                             coords, #coordinates of input data points
#                                             temperature_grid.to_numpy().ravel(), #values of input data points
#                                             (np.indices(lunar_north_pole_img.shape[:2])[1], np.indices(lunar_north_pole_img.shape[:2])[0]), #coordinates of the points where to interpolate -> using the pixels of the lunar image (:2, because cholor channels in 3rd index are not to be considered)
#                                             method='cubic' #as a more complex interpolation method (alt: linear, nearest)
#                                             )
        
#     #     #Potential C-Maps
#     #     colors = ['PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu', 'RdYlBu', 'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic',
#     #             'viridis', 'plasma', 'inferno', 'magma', 'cividis',
#     #             'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds', 'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu', 'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn',
#     #             'binary', 'gist_yarg', 'gist_gray', 'gray', 'bone', 'pink', 'spring', 'summer', 'autumn', 'winter', 'cool', 'Wistia', 'hot', 'afmhot', 'gist_heat', 'copper',
#     #             'twilight', 'twilight_shifted', 'hsv',
#     #             'flag', 'prism', 'ocean', 'gist_earth', 'terrain', 'gist_stern', 'gnuplot', 'gnuplot2', 'CMRmap', 'cubehelix', 'brg', 'gist_rainbow', 'rainbow', 'jet', 'turbo', 'nipy_spectral', 'gist_ncar']

#     #     #Color the North Pole region based on the mapped temperature data
#     #     for color in colors:
#     #         color = 'rainbow'
#     #         colormap = plt.get_cmap(f'{color}') #{color}')
#     #         #colormap = sns.color_palette("viridis", as_cmap=True)
#     #         #colormap = sns.color_palette("Spectral", as_cmap=True)
#     #         #colormap = sns.color_palette("coolwarm", as_cmap=True)
#     #         #colormap = sns.cubehelix_palette(start=0, rot=2, dark=0, light=1, as_cmap=True)

#     #         colored_temps = colormap(temperature_data_mapped)

#     #         #Apply the mask to only color the region of interest
#     #         colored_roi = (colored_temps[:, :, :3] * 255 * (mask[:, :, np.newaxis] / 255)).astype(np.uint8) #the mask-normalization serves the purpose of turning the mask values (0&255) into 0 & 1s -> Multiplying this with the colored_temps "delets" all colors out of the region of interest
#     #         colored_image = cv2.addWeighted(lunar_north_pole_img, 0.4, colored_roi, 0.6, 0) #laying the lunar image and the colors of the region of interests on top of each other 

#     #         #Save the final image
#     #         final_image = cv2.cvtColor(colored_image, cv2.COLOR_BGR2RGB)
#     #         current_image = Image.fromarray(final_image)
#     #         current_image.save(f'../results/Lunar_tbol_Choropleth_c={color}_op={operation}_t=daily.tiff')

#     #     axs[iteration].imshow(current_image)
#     #     axs[iteration].set_title(f'{operation}-Values')
        
#     # p1 = np.percentile(lunar_temp_data_analysed['       tbol',  'mean'], 1)
#     # p99 = np.percentile(lunar_temp_data_analysed['       tbol',  'mean'], 99)

#     # #Create the Normalize object using the min and max percentiles
#     # vmin = p1
#     # vmax = p99
#     # norm = Normalize(vmin=vmin, vmax=vmax)

#     # #Create the ScalarMappable object
#     # sm = ScalarMappable(cmap=colormap, norm=norm)

#     # # Add a colorbar to the plot using the ScalarMappable object
#     # fig.colorbar(sm, orientation='vertical', label='Temp')

#     # fig.suptitle(f'Lunar Bolometric Brightness Temperature Choropleth: Daily \n Lat: 90°N to 80°N Long: 180°W to -180°E; 5-July-2009 to 17-Feb-2019', fontsize=38, ha='center')

#     # plt.savefig('../results/Lunar_tbol_Choropleth_t=daily.png')

#     # plt.show()    # Create a heatmap using plotly's Heatmap trace
#         heatmap = go.Heatmap(
#             x=cart_x.ravel(),  # Provide the x-coordinates
#             y=cart_y.ravel(),  # Provide the y-coordinates
#             z=temperature_data_mapped.ravel(),  # Provide the temperature values
#             colorscale='Rainbow',  # Choose the desired colorscale
#             colorbar=dict(title='Temp')  # Set the colorbar title
#         )

#         # Add the heatmap to the corresponding subplot
#         fig.add_trace(heatmap, row=1, col=i+1)

#     # Update the layout of the figure
#     fig.update_layout(
#         title="Lunar Bolometric Brightness Temperature Choropleth: Daily",
#         xaxis=dict(title='x-axis'),  # Provide the x-axis title
#         yaxis=dict(title='y-axis'),  # Provide the y-axis title
#         showlegend=False  # Hide the legend
#     )

#     return fig

# if __name__ == '__main__':
#     app.run_server(debug=True, port = 5003)



































import dash
from dash import dcc
from dash import html
import plotly.graph_objs as go
from dash.dependencies import Input, Output
import pandas as pd
import io
from functools import reduce
import requests
import numpy as np
import pandas as pd
import geopandas as gpd
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
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import pickle
import pandas as pd
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.subplots as sp
from plotly.tools import FigureFactory as FF
from scipy.interpolate import griddata


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css', 'https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/css/bootstrap.min.css']
import os

print(os.getcwd())


#logo = image_to_base64('../.png')

# Laden Sie Ihre Daten
lunar_temp_data_analysed = pd.read_hdf('../data/lunar_temp_data_analysed_compressed', key='data')
# with open('lunar_temp_data_analysed_compressed', 'r') as f:
#     # your code here


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
        html.Img(src="https://thespringinstitute.com/wp-content/themes/WP-TheSpringInstituteV1/assets/img/astronaut.png", height='50', width='auto', style={'display': 'inline-block', 'vertical-align': 'middle', 'marginLeft': 'auto'}),
#    ], style={'width': 'auto', 'height': 'auto', 'display': 'flex', 'justify-content': 'flex-start', 'align-items': 'center'}),

],
    color="#F5F5F5",
    # style={"background-color": "#242D3D", "padding": "20px"},
    # dark=True
    
)


app = dash.Dash(__name__, external_stylesheets=external_stylesheets)


app.layout = html.Div(
    children=[
        navbar,
        html.Div([
        html.P(id="description",
               children='This application calculates the feasibility score of potential landing site areas for the Forest On The Moon Mission. The feasibility calculation depends on the feature constraint ranges defined in the sliders at the bottom of this app. The higher the normed feasibility score, the better the feasibility. Being an MVP for a holisitc lunar surface analysisi, this a application covers the lunar north pole, respectively Lat: 90°N to 80°N Long: 180°W to -180°E (5-July-2009 to 17-Feb-2019)', style={'font-size': '14px'}),
        ], style={'border-left': '8px solid #346EFD', 'padding-left': '10px', 'margin-top': '60px', 'margin-bottum': '70px',  'margin-left': '50px', 'margin-right': '50px', 'font-size': '10px'}),
        html.Div(
            [
                html.Div(
                    dcc.Graph(
                        id='lunar-temp-graph',
                    ),
                    style={'width': '45%', 'margin-left': '80px', 'margin-top': '30px'}
                ),
                html.Div(
                    [
                        html.Label('Minimum Temperature', style={'font-size': '18px', 'margin-top': '  150px'}),
                        dcc.Slider(150, 350, 1,
                            value=243,
                            id='temp-min-slider',
                            included=True,
                            tooltip={'always_visible': True, 'placement': 'bottom'},
                            marks={i: {'label': '{}°K'.format(i), 'style': {'color': 'white'}} for i in range(150, 351, 20)},
                        ),
                        html.Label('Maximum Temperature', style={'font-size': '18px', 'margin-top': '50px'}),
                        dcc.Slider(
                            id='temp-max-slider',
                            min=150,
                            max=350,
                            value=303,
                            step=1,
                            marks={i: {'label': '{}°K'.format(i), 'style': {'color': 'white'}} for i in range(150, 351, 20)},
                        ),
                        html.Label('Average Temperature', style={'font-size': '18px', 'margin-top':'50px'}),
                        dcc.Slider(
                            id='temp-avg-slider',
                            min=150,
                            max=350,
                            value=283,
                            step=1,
                            marks={i: {'label': '{}°K'.format(i), 'style': {'color': 'white', 'color': 'white'}} for i in range(150, 351, 20)},
                        ),
                        html.Div(
                            style={'backgroundColor': 'blue', 'width': '80%', 'margin': 'auto'}
                        ),
                    ],
                    style={'width': '45%', 'margin-right': '80px', 'margin-left': 'auto'}
                )
            ],
            style={'display': 'flex'}
        ),
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
        'margin': '-8px',
    },
)

@app.callback(
    Output('lunar-temp-graph', 'figure'),
    [Input('temp-min-slider', 'value'),
     Input('temp-max-slider', 'value'),
     Input('temp-avg-slider', 'value')]
)
def update_figure(temp_min, temp_max, temp_avg):


    lunar_temp_data_analysed = pd.read_hdf('../data/lunar_temp_data_analysed_compressed', key='data')

    #Create Visualization with topographic map of the moon surface

    #fig, axs = plt.subplots(1, 3, figsize=(80, 20))
    fig = sp.make_subplots(shared_yaxes=True, subplot_titles=['Min', 'Max', 'Mean'])

    # Read the lunar topography image using Pillow
    lunar_north_pole_img_original = Image.open('../data/6304h.tiff')

    #convert lunar image into numpy array and get it into the right color format for further processing the image with libaries such as OpenCV
    lunar_img_np = np.array(lunar_north_pole_img_original)
    lunar_north_pole_img = cv2.cvtColor(lunar_img_np, cv2.COLOR_RGB2BGR)

    #convert lunar north pole image into greyscale format to have a baseline map for choropleth map
    lunar_north_pole_img_grey = cv2.cvtColor(lunar_north_pole_img, cv2.COLOR_RGB2GRAY)

    #splitting the lunar_north_pole_image binarily into tha part with the black background and the coloured part
    #all pixel colour values under 1 (-> near black values) get the value 0, the others get the value 255
    #based on the number assignements all pixels with the value 255 are selected to create a mask, which is the part of the picture containing the satellite map, excluding the near-black background
    #_ stores the threshold value and is not required anymore
    #mask stores the relevant pixel positions 
    _, mask = cv2.threshold(lunar_north_pole_img_grey, 1, 255, cv2.THRESH_BINARY)

    #Find the center and radius of the North Pole region
    #1. Get identified contours of the picture (outer lines of identied forms)
    contours, _ = cv2.findContours(mask, 
                                cv2.RETR_EXTERNAL, #Taking only the extreme outer contours, because contour hierarchies are not relevant in this case
                                cv2.CHAIN_APPROX_SIMPLE #Removes redundant point and compreses data complexity
                                )

    #2. Determine the maximum area enclosed by the contour using the max-function
    contour_area = max( #determines the maximum elememnt in an iterable 
                    contours, #iterable = list of contours
                    key=cv2.contourArea #key to choose largest elements: cv2.contourArea which determines the area size enclosed by a contour
                    )

    #3. Determine the minimum circle that encloses the contour_area and get center + radius
    (x, y), radius = cv2.minEnclosingCircle(contour_area)

    lunar_temp_data_analysed['feasibility_score'] = lunar_temp_data_analysed['       tbol',  'min'] - temp_min - (temp_max -lunar_temp_data_analysed['       tbol',  'max'] ) - np.sqrt(((temp_avg - (temp_avg - lunar_temp_data_analysed['       tbol',  'mean'])) ** 2))
    #1/(((temp_min - lunar_temp_data_analysed['       tbol',  'min'])** 2) * 1/3 + ((temp_max - lunar_temp_data_analysed['       tbol',  'max']) ** 2) * 1/3 + ((temp_avg - lunar_temp_data_analysed['       tbol',  'mean']) ** 2) * 1/3)

    X = lunar_temp_data_analysed['        clon']
    Y = lunar_temp_data_analysed['       clat']

    Z = lunar_temp_data_analysed['feasibility_score']

    print(Z)

    Z_min = Z.min()
    Z_max = Z.max()
    Z_normalized = (Z - Z_min) / (Z_max - Z_min)

    Z = Z_normalized.copy()

    #Bring data into a coordinate-grid
    temperature_grid = lunar_temp_data_analysed.pivot(index='       clat', columns='        clon', values=('feasibility_score'))
    temperature_grid = temperature_grid.fillna(Z_normalized.mean()).copy()

    #Polar coordinates (clats, clons) of the temperature data
    clats, clons = np.mgrid[80:90:(temperature_grid.shape[0] * 1j), -180:180:(temperature_grid.shape[1] * 1j)]

    #Convert the polar coordinates to Cartesian coordinates 
    radius_scaled = radius * (clats - 80) / 10 #calculated the scaled_radius by dividing the actual clat delta by the max clat delta
    cart_x = radius_scaled * np.cos(np.radians(clons)) + x #radians(clons) converts degrees to radians, by multyplining the radius_scaled by the cos of that radiant the x values can be determined 
                                                            #Imagine triangles in Cartesian coordinate system (one edge along the x-axis, one between the center and the pixel,...)
                                                            #The + x helps to set the centre to the centre of the lunar image 
    cart_y = radius_scaled * np.sin(np.radians(clons)) + y # "

    coords = np.array([cart_x.ravel(), cart_y.ravel()]).T #coordinates are each flattend into 1D arrays, mapped by their index and transposed to get in the right format

    #Data Interpolation to create a dataframe of the same size as the lunar image that contains the brightness values
    temperature_data_mapped = griddata( #griddata performs interpolation - estimating a value based on a set of known data points to, in this case, reduce the amount of data points
                                            coords, #coordinates of input data points
                                            temperature_grid.to_numpy().ravel(), #values of input data points
                                            (np.indices(lunar_north_pole_img.shape[:2])[1], np.indices(lunar_north_pole_img.shape[:2])[0]), #coordinates of the points where to interpolate -> using the pixels of the lunar image (:2, because cholor channels in 3rd index are not to be considered)
                                            method='cubic' #as a more complex interpolation method (alt: linear, nearest)
                                            )
        

    # plt.show()    # Create a heatmap using plotly's Heatmap trace
    heatmap = go.Heatmap(
        x=cart_x.ravel(),  # Provide the x-coordinates
        y=cart_y.ravel(),  # Provide the y-coordinates
        z=temperature_data_mapped.ravel(),  # Provide the temperature values
        colorscale='Rainbow',  # Choose the desired colorscale
        colorbar=dict(title='Temp')  # Set the colorbar title
        )

    # Update the layout of the figure
    fig.update_layout(
        title="Lunar Bolometric Brightness Temperature Choropleth: Daily",
        xaxis=dict(title='x-axis'),  # Provide the x-axis title
        yaxis=dict(title='y-axis'),  # Provide the y-axis title
        showlegend=False  # Hide the legend
    )

    return fig

if __name__ == '__main__':
    app.run_server(debug=True, port = 5003)


































