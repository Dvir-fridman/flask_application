import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import ElasticNet, Lasso
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle
from datetime import datetime
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, LabelEncoder
from predictor import prepare_car_data

data = pd.read_excel('dataset.xlsx')
#נקרא לפונקציה שמנקה את הנתנוים
data = prepare_car_data(data)

df = data.copy()



#Convert categorical variables to numeric
label_encoder = LabelEncoder()
categorical_columns = ['Engine_category','model_category']
for col in categorical_columns:
    df[col] = label_encoder.fit_transform(df[col])

df = pd.get_dummies(df, columns=['District', 'Country_Category'], drop_first=False)
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import ElasticNet, Lasso
from sklearn.metrics import mean_squared_error

#Selected features
X = df.drop(['Price', 'category_color','engine_volume_category','Ownership','Country_Category_קוריאה','is_automatic'], axis=1) #RMSE: 12961.772705838266
y = df['Price']

#Distribution of the data for training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Normalization of the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#We added polynomial variables because from our testing it improves the performance of the linear model by adding new features that represent non-linear relationships between the existing features. Compared to the model without the addition of the polynomial variables that brought us a bigger error
poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly.fit_transform(X_train_scaled)
X_test_poly = poly.transform(X_test_scaled)

lasso = Lasso(alpha=0.01, max_iter=5000, random_state=42)
lasso.fit(X_train_poly, y_train)

# Selecting properties with non-zero coefficients
selected_features = lasso.coef_ != 0
X_train_selected = X_train_poly[:, selected_features]
X_test_selected = X_test_poly[:, selected_features]

#We chose to use "Grid Search", which searches for the best parameters for a machine learning model.
# param_grid = {
#     'alpha': [0.01, 0.1, 1, 10, 100],
#     'l1_ratio': [0.1, 0.5, 0.7, 0.9, 1.0],
#     'max_iter': [1000, 5000, 10000]
# }

# elastic_net = ElasticNet(random_state=42)
# grid_search = GridSearchCV(estimator=elastic_net, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
# grid_search.fit(X_train_selected, y_train)

# Displaying the optimal parameters
best_params = grid_search.best_params_
print(f"Best parameters found: {best_params}")

#Training the model
best_elastic_net = grid_search.best_estimator_
best_elastic_net.fit(X_train_selected, y_train)
print(X_test_selected)
#prediction
y_pred_selected = best_elastic_net.predict(X_test_selected)
pickle.dump(best_elastic_net, open("trained_model.pkl","wb")) 
