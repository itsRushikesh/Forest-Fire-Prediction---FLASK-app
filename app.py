from flask import Flask,render_template,Response,request
import pickle

app = Flask(__name__)

@app.route('/')
def home():
    return"Welcome to home page"

@app.route('/predictData',methods = ['GET','POST'])
def predict_data():
    scalingDown = pickle.load(open('models/scaler.pkl','rb'))
    ridgeAlgo = pickle.load(open('models/ridge.pkl','rb'))

    if request.method == 'POST':
        Tempreature =  float(request.form.get('Tempreature'))
        rh = float(request.form.get('RH'))
        ws = float(request.form.get('Ws'))
        rain = float(request.form.get('Rain'))
        ffmc = float(request.form.get('FFMC'))
        dmc = float(request.form.get('DMC'))
        isi = float(request.form.get('ISI'))
        classes = float(request.form.get('Classes'))
        region = float(request.form.get('Region'))

        newScaledData = scalingDown.transform([[Tempreature,rh,ws,rain,ffmc,dmc,isi,classes,region]])
        result = ridgeAlgo.predict(newScaledData)
        result=result[0]
        print(result)
        return render_template('predict.html',result=result)
    else:
        return render_template('predict.html')

if __name__ == '__main__':
    app.run(port = 8080,debug=True)