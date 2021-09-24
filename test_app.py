from flask import Flask, jsonify, json, request,render_template
from logging import debug
import pickle
from flask_cors import CORS # library for handling cross origin resources sharing.
from predict import predict_data

'''
Ini script untuk dicoba ke POSTMAN. Ga dikasih wrapper.
'''

app = Flask(__name__)
CORS(app)

@app.route("/", methods=["GET"])
def default():
    return render_template("index_test.html") #kurang ini?

@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        data_input = request.get_json()
        data={}
        data["person_age"] = int(data_input['data']['person_age'])
        data["person_income"] = int(data_input['data']['person_income'])
        data["person_home_ownership"] = data_input['data']['person_home_ownership']
        data["person_emp_length"] = float(data_input['data']['person_emp_length'])
        data["loan_intent"] = data_input['data']['loan_intent']
        data["loan_grade"] = data_input['data']['loan_grade']
        data["loan_amnt"] = int(data_input['data']['loan_amnt'])
        data["loan_int_rate"] = float(data_input['data']['loan_int_rate'])
        data["loan_percent_income"] = float(data_input['data']['loan_percent_income'])
        data["cb_person_default_on_file"] = data_input['data']['cb_person_default_on_file']
        data["cb_person_cred_hist_length"] = int(data_input['data']['cb_person_cred_hist_length'])

        result = predict_data(data)

        threshold = 0.5
        risk = True if result >= threshold else False

        result_dict = {
            'model':'credit-risk-predictor',
            'risky':risk,
            'score_proba':float(result[0]),
            'version': '0.4.6',
        }
    return jsonify(result_dict)

if __name__ == '__main__':
    app.run(port=5000, debug=False)
