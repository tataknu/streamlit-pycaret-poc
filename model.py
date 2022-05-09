import os 
from pycaret.regression import *

FILENAME = "results.csv"
MODEL_RESULTS = "/Users/cristianvildosola/Develop/streamlit/"
TRAIN_DATA = "/Users/cristianvildosola/Downloads/data.csv"

# Loading data
df = pd.read_csv(TRAIN_DATA)

# Setting target column for prediction
s = setup(df, target = 'Score', session_id=123)

# Creating and comparing models + tunning
best = compare_models()

exp_clf102 = (setup(
	data = df,
	target = 'Score',
	session_id=123,
	normalize = True,
	transformation = True))

et_model = create_model('et')

# Saving...
results = pull()
pd.to_csv(MODEL_RESULTS+FILENAME, index=False)
save_model(et_model, model_name = 'extra_tree_model_2')