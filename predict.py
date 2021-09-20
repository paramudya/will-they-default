import pandas as pd
import pickle
import numpy as np
import xgboost

#format input sekalian buat testing
raw_input={'person_age': 30,
  'person_income': 82000,
  'person_home_ownership': 'MORTGAGE',
  'person_emp_length': 5.0,
  'loan_intent': 'PERSONAL',
  'loan_grade': 'A',
  'loan_amnt': 5000,
  'loan_int_rate': 6.76,
  'loan_percent_income': 0.06,
  'cb_person_default_on_file': 'N',
  'cb_person_cred_hist_length': 8,
#   'loan_status_predict': 0,
#   'loan_status_proba': 0.013454122468829155
  }

def predict_data(raw_input):
    with open('saved_models/pipe_to_deploy.pkl', 'rb') as f_in: 
        preprocess= pickle.load(f_in)

    with open('saved_models/xgb_to_deploy.pkl', 'rb') as f: 
        model= pickle.load(f)

    raw_processed = pd.DataFrame.from_dict(raw_input, orient='index').T.replace({
        None: np.nan,
        "null":np.nan,
        "" : np.nan
    })
    X = preprocess.transform(raw_processed)
    result = model.predict_proba(X)[:,1]
    return result
    
if __name__ == "__main__":
    predicted=predict_data(raw_input)

    print(type(predicted))
    print(predicted)
    # assert score_proba == result
