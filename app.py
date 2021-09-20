from flask import Flask, jsonify, json, request, render_template
from logging import debug
import pickle
from flask_cors import CORS # library for handling cross origin resources sharing.
from predict import predict_data

def new_app():
    app = Flask(__name__)
    CORS(app)

    @app.route("/", methods=['GET'])
    def hello():
        return render_template("index.html")

    @app.route("/predict", methods=['POST'])
    def opredict():
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
    return app

# if __name__ == '__main__':
#     app.run(port=5000, debug=False)
