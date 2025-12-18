from flask import Flask,jsonify,request,render_template
import pandas as pd
import pickle

app=Flask(__name__)


def get_cleaned_data(data):
    print(dict(data))
    data=dict(data)
    data["gestation"]=[float(data["gestation"])]
    data["age"]=[float(data["age"])]
    data["parity"]=[int(data["parity"])]
    data["smoke"]=[float(data["smoke"])]
    data["height"]=[float(data["height"])]
    data["weight"]=[float(data["weight"])]
    # print(data)
    return data
    # return "cnosdjds"

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict",methods=["post"])
def predict():
    baby_data_form=request.form
    baby_data=get_cleaned_data(baby_data_form)
    baby_df=pd.DataFrame(baby_data)
    # return " yooo"
    print(baby_df)
    with open("model/model.pkl","rb") as f:
        mymodel=pickle.load(f)
        
    predicted_data=mymodel.predict(baby_df)
    
    response=round(float(predicted_data),2)
    return render_template("index.html",bwt=response)


if __name__=="__main__":
    app.run(debug=True)
    