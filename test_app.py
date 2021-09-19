from flask import Flask, jsonify, json, request
from logging import debug
import pickle
from flask_cors import CORS # library for handling cross origin resources sharing.
from predict import predict_data

'''
Ini app.py yang untuk dicoba ke POSTMAN. Ga dikasih wrapper.
'''

app = Flask(__name__)
CORS(app)

@app.route("/predict", methods=['POST'])
def do_predict():
    if request.method == 'POST':
        data_input = request.get_json()
        print('Input data:\n',data_input,'\nwith type:',type(data_input))
        result = predict_data(data_input)
        
        threshold = 0.5
        risk = True if result >= threshold else False
        
        result_dict = {
            'model':'credit-risk-scorer-dari-modul-agustiar',
            'risky':risk,
            'score_proba':float(result[0]),
            'version': '1.0.Ordinal-punya-sendiri',

        }
        print('Proba of risk: ',result)
    return jsonify(result_dict)

if __name__ == '__main__':
    app.run(port=5000, debug=False)

'''
input format:

 {
     "person_age": 26,
  "person_income": 48000,
  "person_home_ownership": "RENT",
  "person_emp_length": 2.0,
  "loan_intent": "MEDICAL",
  "loan_grade": "B",
  "loan_amnt": 10000,
  "loan_int_rate": 12.21,
  "loan_percent_income": 0.21,
  "cb_person_default_on_file": "N",
  "cb_person_cred_hist_length": 3
 }
 
expected test proba result from above input: .03851019963622093


expected output format:
{
                'model':'credit-risk-scorer-dari-modul-agustiar',
                'risky':False,
                'score_proba':.03851019963622093,
                'version': '1.0.Ordinal-punya-sendiri',

            }

http to access from(?):
http://127.0.0.1:5000/predict
with method POST
'''