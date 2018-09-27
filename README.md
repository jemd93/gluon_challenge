# Data augmentation via random utterance generation

This repository contains the code to train a model that learns from 
the utterance text of a given intent and is able to generate random utterances
based on what it learned.

# Repo Summary
- `checkpoints` - Checkpoint files that the model automatically creates while training
- `data` - Folder to store all the data used to train/test the models
- `model_params` - Txt files with combinations of model parameters
- `MODELS` - Here is where all models get saved to
- `model_scripts` - This folder contains any scripts used to train/test/optimize the models
- `get_atlas_data.py` - Downloads data from atlas and saves it to a csv and pickle file
- `train_model.sh` - Will train a model using a parameters file inside of `model_params` folder
- `test_model.sh` - Will generate 1000 characters using a model inside of `MODELS`

# Downloading data from atlas

`python get_atlas_data.py --output_file=name_of_data_file`

This will download all english utterances with confirmation status confirmed 
from atlas and save them as a pickle and csv file in the `./data` folder.

# Training a model

`bash train_model.sh name_of_param_file input_data_file output_data_file model_name intent_name`

- `name_of_param_file` - Name of parameter file inside of the `model_params` folder
- `input_data_file` - Name of the data file with .pickle extension
- `output_data_file` - Name of the output data file where the intent-specific filtered data will be stored
- `model_name` - Name of the model to save
- `intent_name` - Intent to train the model with

Trains the model with data of a given intent. After training, the model should be able
to generate random utterances of that intent.

# Test the model

`bash test_model.sh name_of_param_file input_data_file model_name 'test_string'`

- `name_of_param_file` - Name of parameter file inside of the `model_params` folder
- `input_data_file` - Name of the filtered data file of a given intent (used to train the model originally)
- `model_name` - Name of the saved model
- `test_string` - String to test the model with (don't forget the single quotes)

Tests the model with a string. The model will try to complete the string and use 1000 characters
to write as many utterances as it can. Meaning if the input string is `'hello I'd like to'` the
model will attempt to finish that utterance and continue to make more new ones until it generates 1000
characters.


