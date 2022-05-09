import os 
from pycaret.regression import *

FILENAME = "results.csv"
MODEL_RESULTS = "/Users/cristianvildosola/Develop/streamlit/"
TRAIN_DATA = "/Users/cristianvildosola/Downloads/data.csv"

# Loading data
df = pd.read_csv(TRAIN_DATA)
df = df[['Score', 'Age', 'Balance', 'Salary']]

# Setting target column for prediction
s = setup(df, target = 'Score', session_id=123)

# Creating and comparing models + tunning
best = compare_models(n_select = 10)

print(best)

best = automl(optimize = 'MAE')

# Saving...
results = pull()
results.to_csv(MODEL_RESULTS+FILENAME)
save_model(best, model_name = 'blended-model-01')
