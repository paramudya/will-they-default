from flask import Flask, jsonify, json, request, render_template
from logging import debug
import pickle
from flask_cors import CORS # library for handling cross origin resources sharing.
from predict import predict_data

def new_app():
    app = Flask(__name__)
    CORS(app)

    @app.route("/", methods=['GET'])
    def default():
        return render_template("index.html")

    @app.route("/predict", methods=['POST'])
    def predict():
        if request.method == 'POST':
            data_input = request.get_json()

            data["person_age"] = data_input.get("person_age")
            data["person_income"] = data_input.get("person_income")
            data["person_home_ownership"] = data_input.get("person_home_ownership")
            data["person_emp_length"] = data_input.get("person_emp_length")
            data["loan_intent accounts"] = data_input.get("loan_intent")
            data["loan_grade"] = data_input.get("loan_grade")
            data["loan_amnt"] = data_input.get("loan_amnt")
            data["loan_percent_income"] = data_input.get("loan_percent_income")
            data["cb_person_default_on_file"] = data_input.get("cb_person_default_on_file")
            data["cb_person_cred_hist_length"] = data_input.get("cb_person_cred_hist_length")
            
            result = predict_data(data)
            
            threshold = 0.5
            risk = True if result >= threshold else False
            
            result_dict = {
                'model':'credit-risk-scorer-dari-modul-agustiar',
                'risky':risk,
                'score_proba':float(result[0]),
                'version': '1.0.Ordinal-punya-sendiri',
            }
        return jsonify(result_dict)
    return app

# if __name__ == '__main__':
#     app.run(port=5000, debug=False)
