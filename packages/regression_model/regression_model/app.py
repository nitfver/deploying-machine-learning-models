import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle


import numpy as np
import pandas as pd

from regression_model.processing.data_management import load_pipeline
from regression_model.config import config
from regression_model.predict import make_prediction
from regression_model.processing.validation import validate_inputs
from regression_model import __version__ as _version

import logging
import typing as t




app = Flask(__name__)
#model = pickle.load(open('model.pkl', 'rb'))


_logger = logging.getLogger(__name__)

pipeline_file_name = f"{config.PIPELINE_SAVE_FILE}{_version}.pkl"
print(pipeline_file_name)
_price_pipe = load_pipeline(file_name=pipeline_file_name)


two_dimensional_list = [config.DATASET_DIR]

#@app.route('/')
#def home():
#    return render_template('index.html',two_dimensional_list=two_dimensional_list)




@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    #int_features = [(x) for x in request.form.values()]

    if request.method == 'POST':
        df = pd.read_csv(request.files.get('file'))
    
    #int_features = request.form.values()
    #final_features = [np.array(int_features)]
    
    #int_features= "".join(int_features)
    #print(int_features)


    #prediction= make_prediction(config.DATASET_DIR/'atest.csv')
    prediction= make_prediction(df)

    #prediction = model.predict(final_features)

    #output = round(prediction[0], 2)
    output= [round(i,2) for i in prediction]


    return render_template('index.html', prediction_text='Estimated House Price should be around $ {}'.format(output))




if __name__ == "__main__":
    app.run(debug=True)