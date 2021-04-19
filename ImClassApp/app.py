#https://towardsdatascience.com/building-a-brain-tumor-classification-app-e9a0eb9f068
#https://stackoverflow.com/questions/65556543/using-dash-to-process-an-image-images-and-running-it-through-a-trained-and-saved
import dash
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html
from dash import no_update
import dash_bootstrap_components as dbc
import base64
import io
from io import BytesIO
import re
from tensorflow import keras
from skimage.transform import resize
import tensorflow as tf

from PIL import Image
import numpy as np


file='my_h5_model.h5'
model = keras.models.load_model(file)
class_names=['Bear', 'Cat', 'Chicken', 'Cow', 'Deer', 'Dog', 'Duck', 'Eagle', 'Elephant', 'Human', 'Lion', 'Monkey', 'Mouse', 'Nat', 'Panda', 'Pig', 'Pigeon', 'Rabbit', 'Sheep', 'Tiger', 'Wolf']

BS = "https://stackpath.bootstrapcdn.com/bootswatch/4.5.2/sketchy/bootstrap.min.css"
app = dash.Dash(external_stylesheets=[BS])

server = app.server

app.layout = html.Div([
    
    html.H1(children='ANIMAL IMAGE CLASSIFICATION', style={'textAlign': 'center'
        }),
    
    
    html.Div([dcc.Markdown('''
                ###### Step 1: Upload a single image with a .jpg or .jpeg format
                ###### Step 2: Wait for predicted label
    ''')], style={'marginLeft': 10}
    ),
    
    dcc.Upload(
        id='upload-image',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files')
        ]),
        style={
            'width': '95%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        }, 
        # Allow multiple files to be uploaded
        multiple=True
    ),

    html.Div(id='output-image-upload', style={'position':'absolute', 'left':'200px', 'top':'250px'}),
    
    html.Div(id='prediction', style={'position':'absolute', 'left':'800px', 'top':'310px', 'font-size':'x-large'}),

])

def parse_contents(contents):
    card  =  dbc.Card(
    [
        dbc.CardImg(src=contents, top=True, style={'height':'297px', 'width':'297px'}),
        dbc.CardBody(
            [
                html.H4(id='prediction', className="card-title")
            ]
        ),
    ],
    style={'left':'300px','height':'300px', 'width':'300px'},
    )
    
    return card


@app.callback([Output('output-image-upload', 'children'), Output('prediction', 'children')],
              [Input('upload-image', 'contents')])

def update_output(list_of_contents):        
    
    if list_of_contents is not None:
        children = parse_contents(list_of_contents[0]) 
         
        img_data = list_of_contents[0]
        img_data = re.sub('data:image/jpeg;base64,', '', img_data)
        img_data = base64.b64decode(img_data)  
        
        stream = io.BytesIO(img_data)
        img_pil = Image.open(stream)
        
        
        #Load model, change image to array and predict
        model = keras.models.load_model(file) 
        dim = (150, 150)
        
        img = np.array(img_pil.resize(dim))
        
        x = img.reshape(1,150,150,3)
       

        ans = model.predict(x)

        pred=class_names[np.argmax(ans)]  
        

        
        return children, pred
    
    else:
        return (no_update, no_update)  

if __name__ == '__main__':
    app.run_server(host='127.0.0.1', port='8050', debug=True)