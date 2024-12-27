import pickle
from flask import Flask,request,app,jsonify,url_for,render_template,redirect,flash
import numpy as np
import pandas as pd
app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello, World!'

if __name__ == '__main__':
    app.run(debug=True)

app = Flask(__name__)
# load the model
model = pickle.load(open('knn_model.pkl','rb'))
# scaler = pickle.load(open('scaler_model.pkl','rb'))
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api',methods=['POST'])

def predict_api():
    data = request.json['data']
    print(data)
    reshaped_data = np.array(list(data.values())).reshape(1, -1)
    # new_data = scaler.transform(np.array(list(data.values())).reshape(1,-1))
    output = model.predict(reshaped_data)
    print(output[0])
    output_int = int(output[0])
    return jsonify({'prediction': output_int})
    # return jsonify(output_int[0])

@app.route('/predict',methods=['POST'])
def predict():
    data=[float(x) for x in request.form.values()]
    final_input = np.array(data).reshape(1,-1)
    # final_input=scalar.transform(np.array(data).reshape(1,-1))
    # print(final_input)
    output=model.predict(final_input)[0]
    return render_template("home.html",prediction_text="Diabetic Prediction {}".format(output))

if __name__=='__main__':
    app.run(debug=True)



