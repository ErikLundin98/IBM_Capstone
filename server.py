from flask import Flask, request
from pickle import load # to load the scaler
import numpy as np

app = Flask(__name__)


scaler = load(open('model/scaler.pkl', 'rb'))
model = load(open('model/final_simple.sav', 'rb'))
print(model)
print('model loaded!')


@app.route('/')
def hello():
    print('someone reached our server')
    return 'hello user! please send a response at the /predict path instead'

@app.route('/predict', methods=['POST'])
def predict():
    if request.method =='POST':
        print('someone reached our server and wanted a prediction')
        data = eval(request.get_json())
        features = data['Data']
        features = [int(feature) for feature in features]
        features = scaler.transform([features]) # scale the same way we did for our dataset
        features = features.tolist()
        features[0].insert(0, 1) # add the bias term
        # Now we can use our model to predict
        print(features)
        prediction = model.predict(features)
        print(prediction)
        return 'Hi {}'.format(data['Name']) + \
        ('! You should get examined right now!' if np.round(prediction) == 1 else ', you are healthy and don\'t need to be examined')



if __name__ == '__main__':
    print('running app...')
    app.run()
