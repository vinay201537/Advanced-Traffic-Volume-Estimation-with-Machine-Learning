import os
import pickle

import numpy as np
import pandas
from flask import Flask, render_template, request

app = Flask(__name__, template_folder="templates")
model = pickle.load(open(r"./model.pkl", "rb"))


@app.route("/")  # route to display the home page
def index():
    return render_template("index.html")  # rendering the home page


# route to show the predictions in a web UI
@app.route("/predict", methods=["POST", "GET"])
def predict():
    #  reading the inputs given by the user
    input_feature = [float(x) for x in request.form.values()]
    features_values = [np.array(input_feature)]
    names = [
        [
            "holiday",
            "temp",
            "rain",
            "snow",
            "weather",
            "year",
            "month",
            "day",
            "hours",
            "minutes",
            "seconds",
        ]
    ]
    data = pandas.DataFrame(features_values, columns=names)
    # predictions using the loaded model file
    prediction = model.predict(data)
    print(prediction)
    text = "Estimated Traffic Volume is :"
    return render_template("output.html", result=text + str(prediction) + "units")

    # showing the prediction results in a UI


if __name__ == "__main__":
    # app.run(host='0.0.0.0', port=8000,debug=True)    # running the app
    port = int(os.environ.get("PORT", 5000))
    app.run(port=port, debug=True, use_reloader=False)
