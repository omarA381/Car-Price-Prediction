import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression,Lasso,Ridge
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score,KFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
plt.style.use('bmh') 
st.set_page_config(page_title="Car Data Dashboard", layout="wide")
# قراءة البيانات
df = pd.read_csv('\numeric_dataset_car.csv')
df.drop(columns=['Unnamed: 0', 'Car_Name'], inplace=True)

car = pd.read_csv('car_prediction_data (1).csv')
# عرض شكل البيانات

# القائمة الجانبية
menu = st.sidebar.radio("Menu", ['Home', 'Predictions'])

# الصفحة الرئيسية
if menu == 'Home':
    #st.title('Exploratory Data Analysis')
    st.markdown("<h1 style='text-align: center;'>Exploratory Data Analysis</h1>", unsafe_allow_html=True)

    # الملخص الإحصائي للبيانات
    st.markdown("<h6 style='text-align: left;'>Summary Statistics</h6>", unsafe_allow_html=True)
    st.table(df.describe())

    #the correlation coefficients between 'Selling_Price' and other numerical features
    st.markdown("<h6 style='text-align: left;'>correlation between Selling Price and other </h6>", unsafe_allow_html=True)
    df_corr = df.corr() 
    st.table(df_corr['Selling_Price'])
    
    graphs_col, tables_col = st.columns(2)


    # الرسوم البيانية
    st.markdown("<h1 style='text-align: center;'>Graphs</h1>", unsafe_allow_html=True)
    df_fuel = car.Fuel_Type.value_counts().rename_axis('Fuel').reset_index(name='count')
    df_seller = car.Seller_Type.value_counts().rename_axis('Seller Type').reset_index(name='count')
    df_Transmission = car.Transmission.value_counts().rename_axis('Transmission').reset_index(name='count')

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
    axes[0].bar(df_fuel['Fuel'], df_fuel['count'])
    axes[0].set_title('Type of Fuel')
    
    axes[1].bar(df_seller['Seller Type'], df_seller['count'])
    axes[1].set_title('Type of Seller')
    
    axes[2].bar(df_Transmission['Transmission'], df_Transmission['count'])
    axes[2].set_title('Type of Transmission')
    st.pyplot(fig)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
    sns.boxplot(x=df['Selling_Price'], ax=axes[0])
    axes[0].set_title('Distribution of Selling Price')

    sns.boxplot(x=df['Present_Price'], ax=axes[1])
    axes[1].set_title('Distribution of Present Price')

    sns.boxplot(x=df['Kms_Driven'], ax=axes[2])
    axes[2].set_title('Distribution of Kms Driven')

    st.pyplot(fig)
    
    
X = df.drop(['Selling_Price'], axis=1)
y = df['Selling_Price']
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
models = {'lr':LinearRegression(),
          'lasso':Lasso(),
          'ridge':Ridge(),
          'Decision Tree Regressor':DecisionTreeRegressor()
          ,'Support Vector Regression':SVR()
          ,'Random Forest Regressor':RandomForestRegressor(),
          'Gradient Boosting Regressor':GradientBoostingRegressor()}


results = []
for model in models.values():
    kf = KFold(n_splits=5,shuffle=True,random_state=42)
    cv_result = cross_val_score(model,X,y,cv=kf)
    results.append(cv_result)
    
    
#--------------------------------------------------------------------
GBR = GradientBoostingRegressor()
kf = KFold(n_splits=5,shuffle=True,random_state=42)
GBR = cross_val_score(GBR,X,y,cv=kf)



if menu == 'Predictions':
    st.markdown("<h1 style='text-align: center;'>Machine Learning Model</h1>", unsafe_allow_html=True)
    fig7 = plt.figure(figsize=(20,8))
    plt.boxplot(results, labels=models.keys())
    st.pyplot(fig7)
    st.write('Gradient Boosting Regressor Test Set Accuracy: 0.9717546778683558')
    f1 = st.text_input('Enter Year',)
    f2 = st.text_input('Enter Present Price')
    f3 = st.text_input('Enter Kms Driven')
    f4 = st.text_input('Enter Fuel Type')
    f5 = st.text_input('Enter Seller Type')
    f6 = st.text_input('Transmission')
    f7 = st.text_input('Enter Owner')
    predict_list = []
    if all([f1, f2, f3, f4, f5, f6, f7]):
    # إنشاء قائمة التنبؤ
       predict_list = [int(f1), float(f2), int(f3), f4, f5, f6, f7]
    but = st.button('Predict')
    if but:
        
            
    # إنشاء نموذج GBR
        GBR = GradientBoostingRegressor()

        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(GBR, X, y, cv=kf)

    # التنبؤ باستخدام النموذج الذي تم تدريبه
        prediction = GBR.fit(X, y).predict([predict_list])

        st.markdown(f"<h1 style='text-align: center;'>{prediction}</h1>", unsafe_allow_html=True)
        


    



    
    
