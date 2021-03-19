import numpy as np
import pandas as pd
from csv import reader

from regression_model.processing.data_management import load_pipeline
from regression_model.config import config
from regression_model.processing.validation import validate_inputs
from regression_model import __version__ as _version

import logging
import typing as t


_logger = logging.getLogger(__name__)

pipeline_file_name = f"{config.PIPELINE_SAVE_FILE}{_version}.pkl"
print(pipeline_file_name)
_price_pipe = load_pipeline(file_name=pipeline_file_name)



#def make_prediction(*, input_data: t.Union[pd.DataFrame, dict],
#                    ) -> dict:

def make_prediction(input_data):
    """Make a prediction using a saved model pipeline.

    Args:
        input_data: Array of model prediction inputs.

    Returns:
        Predictions for each input row, as well as the model version.
    """
    #data= pd.read_csv(input_data)
    data = input_data

    validated_data = validate_inputs(input_data=data)
    print("validated data", validated_data)

    prediction = _price_pipe.predict(validated_data[config.FEATURES])

    print("PREDICTION",prediction)
    output = np.exp(prediction)

    #results = {"predictions": output, "version": _version}
    results = output
    _logger.info(
        f"Making predictions with model version: {_version} "
        f"Inputs: {validated_data} "
        f"Predictions: {results}"
    )

    return results

#print(config.DATASET_DIR)


#result= make_prediction(config.DATASET_DIR/'atest.csv')
#print(result)