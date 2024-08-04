import numpy as np
from flask import Flask, request, jsonify, render_template
import pandas as pd
import pickle
import os

from sklearn.discriminant_analysis import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures
from car_data_prep import prepare_data

app = Flask(__name__)
modelCar = pickle.load(open("trained_model.pkl","rb"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
        data = request.form.to_dict()
        data.update({
        'Repub_date': None,
        'Description': None,
        'Test': None,
        'Pic_num': None,
        'Supply_score': None,
        'Price': np.nan
        })
        data['capacity_Engine'] = int(data['capacity_Engine'])
        data['Year'] = int(data['Year'])
        data['Hand'] = int(data['Hand'])
        data['Km'] = int(data['Km'])
        # prev_ownership = None
        # test= None
        # cre_date = None
        # repub_date= None
        # Description= None
        # Pic_num= None
        # Supply_score= None
        # Area= None
        # Price=np.nan
        df = pd.DataFrame(data, index=[0])
        df_prepared = prepare_data(df)
        df_prepared.drop(columns=['Price'])

        df = df_prepared.copy()
    #Convert categorical variables to numeric
        label_encoder = LabelEncoder()
        categorical_columns = ['Engine_category','model_category']
        for col in categorical_columns:
            df[col] = label_encoder.fit_transform(df[col])

        df = pd.get_dummies(df, columns=['District', 'Country_Category'], drop_first=False)

        if df.shape[0] == 1:
            columns_to_check = [
                'District_ב"ש והדרום', 'District_חיפה וצפון', 'District_ירושלים', 'District_ת"א והמרכז',
                'Country_Category_אחר', 'Country_Category_ארה"ב', 'Country_Category_גרמניה',
                'Country_Category_יפן', 'Country_Category_צכיה', 'Country_Category_צרפת', 'Country_Category_קוריאה'
            ]
            
            for column in columns_to_check:
                if column not in df.columns:
                    df[column] = False

        X = df.drop(['Price','category_color','engine_volume_category','Ownership','Country_Category_קוריאה','is_automatic'], axis=1) 

        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        X = scaler.transform(X)

        poly = PolynomialFeatures(degree=2, include_bias=False)
        X = poly.fit_transform(X)
        X = X[:, :128]
        prediction = modelCar.predict(X)
        return f'מחיר הרכב החזוי: {prediction[0]:.2f}'

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))    
    app.run(host='0.0.0.0', port=port,debug=True)
