from flask import Flask, request, jsonify, render_template
from flask_cors import CORS # library for handling cross origin resources sharing.
from predict import make_predictions, preprocess_input


def create_app():
    """ app factories """
    app = Flask(__name__)
    CORS(app)


    @app.route("/", methods=["GET"])
    def default():
        return render_template("index.html")


    @app.route('/predict', methods=['POST'])
    def predict():
        if request.method == 'POST':
            data_input = request.get_json()["data"]
            data = {}

            data["Age"] = data_input.get("Age")
            data["Sex"] = data_input.get("Sex")
            data["Job"] = data_input.get("Job")
            data["Housing"] = data_input.get("Housing")
            data["Saving accounts"] = data_input.get("saving_account")
            data["Checking account"] = data_input.get("checking_account")
            data["Credit amount"] = data_input.get("credit_amount")
            data["Duration"] = data_input.get("duration")
            data["Purpose"] = data_input.get("purpose")


            result = make_predictions(data)
            
            result = {
                "model_version": "german_credit_1.0.0",
                "api_version": "v1",
                "result": str(round(list(result)[0], 3))
            }
            
        return jsonify(result)
    return app
