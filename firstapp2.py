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
import plotly.express as px
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
df = pd.read_csv('D:\\Streamlit\\numeric_dataset_car.csv')
df.drop(columns=['Unnamed: 0', 'Car_Name'], inplace=True)

car = pd.read_csv('D:\\Streamlit\\car_prediction_data (1).csv')
# عرض شكل البيانات

# القائمة الجانبية
menu = st.sidebar.radio("Menu", ['Home', 'Predictions','About me'])

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
    


    selected_chart = st.selectbox('Select Chart', [None, 'Bar', 'Scatter', 'BoxPlot','Heatmap'])

    if selected_chart =='Bar':
        selected_column = st.selectbox('select category column', car.columns[car.dtypes == 'O'])
        value_counts = car[selected_column].value_counts()
        fig = px.bar(x=value_counts.index, y=value_counts.values, labels={selected_column: 'القيمة', 'index': 'القيمة'})
        st.plotly_chart(fig)
    elif selected_chart == 'Scatter':
        numeric_columns =  list(car.select_dtypes(include=['int', 'float']).columns)
        x = st.selectbox('X-Axis', numeric_columns)
        y = st.selectbox('Y-Axis', numeric_columns)
        fig2 = px.scatter(df, x=x, y=y)
        st.plotly_chart(fig2)
    elif selected_chart =='BoxPlot':
        numeric_columns =  list(car.select_dtypes(include=['int', 'float']).columns)
        col = st.selectbox('Columns', numeric_columns)
        fig3 = px.histogram(car,x=col)
        st.plotly_chart(fig3)
    elif selected_chart =='Heatmap':
        fig4 = px.imshow(df.corr(), text_auto=True)
        fig4.update_layout(
        autosize=False,
        width=1000,  # ضبط عرض الرسم البياني حسب الحاجة
        height=800,) # ضبط ارتفاع الرسم البياني حسب الحاجة)
        st.plotly_chart(fig4)

    

if menu == 'Predictions':
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
        GBR = GradientBoostingRegressor()
        kf = KFold(n_splits=5,shuffle=True,random_state=42)
        GBR = cross_val_score(GBR,X,y,cv=kf)

    
    st.markdown("<h1 style='text-align: center;'>Car Price Prediction</h1>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center;'>Gradient Boosting Regressor</h4>", unsafe_allow_html=True)
    #Year = st.slider("Pick a Year", 2013, 2018)
    #Present_Price = st.slider("Pick a Present Price	", 0.32, 92.600)
    #Kms_Driven = st.slider("Pick a Kms Driven", 500, 500000)
    #Fuel_Type = st.radio("Pick a Fuel Type", ['0','1','2'])
    #Seller_Type = st.radio("Pick a SellerType",['0','1'])
    #Transmission = st.radio("Pick a Transmission",['0','1'])
    #Owner = st.slider("Pick a Owner",0,3)
    # Organize them horizontally
    st.write("### Select Options:")

# Organize them horizontally
    col1, col2, col3,col4 = st.columns(4)

# Column 1: Fuel Type
    with col1:
        Fuel_Type = st.radio("Pick a Fuel Type", ['0', '1', '2'])

# Column 2: Seller Type
    with col2:
        Seller_Type = st.radio("Pick a SellerType", ['0', '1'])

# Column 3: Transmission
    with col3:
        Transmission = st.radio("Pick a Transmission", ['0', '1'])
    with col4:
        Owner = st.radio("Pick a Owner",['0','1','3'])
        
    col_1,col_2,col_3 = st.columns(3)
    with col_1:
        Present_Price = st.slider("Pick a Present Price	", 0.32, 92.600)
    with col_2:
        Kms_Driven = st.slider("Pick a Kms Driven", 500, 500000)
    with col_3:
        Year = st.slider("Pick a Year", 2013, 2018)
        
    user_inputs = np.array([Year, Present_Price, Kms_Driven, Fuel_Type, Seller_Type, Transmission, Owner])   
        
    
    #user_inputs = np.array([Year, Present_Price, Kms_Driven, Fuel_Type, Seller_Type, Transmission, Owner])
    but = st.button('Predict')
    if but: 
        GBR = GradientBoostingRegressor()
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(GBR, X, y, cv=kf)
        prediction = GBR.fit(X, y).predict([user_inputs])
        st.markdown(f"<h1 style='text-align: center;'>{prediction}</h1>", unsafe_allow_html=True)
if menu == 'About me':
    st.title("My Personal Information")
    st.markdown("<h6 style='text-align: left;'>Name: Omar Ahmed Badr </h6>", unsafe_allow_html=True)
    st.markdown("<h6 style='text-align: left;'>Email: <a href='mailto:badromar028@gmail.com'>badromar028@gmail.com</a> </h6>", unsafe_allow_html=True)
    st.markdown("<h6 style='text-align: left;'>LinkedIn: <a href='https://www.linkedin.com/in/omar-badr-762b07258/'>https://www.linkedin.com/in/omar-badr-762b07258/</a> </h6>", unsafe_allow_html=True)
    st.markdown("<h6 style='text-align: left;'>github: <a href='https://github.com/omarA381 '>https://github.com/omarA381</a> </h6>", unsafe_allow_html=True)
    st.markdown("<h6 style='text-align: left;'>Phone: +20 1207371920</h6>", unsafe_allow_html=True)

    

