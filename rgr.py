import pandas as pd
import streamlit as st
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb
import tensorflow as tf
import pickle
import random

import sklearn as sk
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

def prepare_data():
    X = pd.read_csv('mumbai_houses_task.csv', on_bad_lines='skip')
    number_names = X.select_dtypes(include='number').columns
    cat_names = X.select_dtypes(include='object').columns
    X_real_mean = X[number_names].fillna(X[number_names].mean())
    X_cat = X[cat_names].fillna('UND')
    X = pd.concat([X_real_mean, X_cat], axis=1)
    X['price'] = X['price']/1000000
    X.rename(columns={'price':'price(mln)'}, inplace='True')
    train, test = train_test_split(X, test_size=0.05, shuffle=True)
    test = pd.concat((test, train.loc[train['Status']=='Under Construction']), axis=0, ignore_index=True)
    train.to_csv('mumbai_houses_task_edit_train.csv', index=False)
    test.to_csv('mumbai_houses_task_edit_test.csv', index=False)

def prepare_models():
    X, y = load_train_data()
    X = pd.get_dummies(X)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pf = PolynomialFeatures(degree=2)
    X_poly = pf.fit_transform(X_scaled)
    params = {'alpha': np.arange(0.1,1, 0.1)}
    ridge_opt = GridSearchCV(Ridge(), params).fit(X_poly, y)
    ridge_poly = Ridge(alpha=ridge_opt.best_params_['alpha'])
    ridge_poly.fit(X_poly, y)

    xgbreg = xgb.XGBRegressor()
    xgbreg.fit(X_scaled, y)

    nn = tf.keras.Sequential([
        tf.keras.layers.Dense(32, activation="relu", input_shape=(19,)),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(1, activation="linear"),
    ])
    nn.summary()
    nn.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.005), loss="mse")
    nn.fit(X_scaled, y, epochs=5)

    pickle.dump(ridge_poly, open('PolynomialRidge.pkl', 'wb'))
    pickle.dump(pf, open('PolynomialFeatures.pkl', 'wb'))
    pickle.dump(scaler, open('StandardScaler.pkl', 'wb'))
    pickle.dump(xgbreg, open('xgbreg.pkl', "wb"))
    pickle.dump(nn, open('NN.pkl', "wb"))

def load_train_data():
    X = pd.read_csv('mumbai_houses_task_edit_train.csv', on_bad_lines='skip')
    y = X['price(mln)']
    X = X.drop(['price(mln)'], axis=1)
    return (X, y)

def load_test_data():
    X = pd.read_csv('mumbai_houses_task_edit_test.csv', on_bad_lines='skip')
    y = X['price(mln)']
    X = X.drop(['price(mln)'], axis=1)
    return (X, y)

def load_models():
    ridge_poly = pickle.load(open('PolynomialRidge.pkl', 'rb'))
    pf = pickle.load(open('PolynomialFeatures.pkl', 'rb'))
    scaler = pickle.load(open('StandardScaler.pkl', "rb"))
    xgbreg = pickle.load(open('xgbreg.pkl', "rb"))
    nn = pickle.load(open('NN.pkl', "rb"))
    return (ridge_poly, pf, scaler, xgbreg, nn)

def info():
    st.title('Информация')
    st.write('Датасет описывает цены на жилье в Мумбаи. Признаками являются площадь, количество разнообразных комнат, наличие удобств \
    вроде лифта, парковки и так далее. Для избавления от операций над слишком большими числами цена теперь вместо просто денежных единиц \
    указывается в миллионах денежных единиц.')
    X = pd.read_csv('mumbai_houses_task.csv')
    st.write(X)


def visualize():
    st.title('Визуализация')
    X, y = load_train_data()
    concat = pd.concat([X,y], axis='columns')
    concat = concat.select_dtypes(include='number')
    fig = plt.figure()
    fig, ax = plt.subplots()
    sns.heatmap(concat.corr(), annot=True)
    st.pyplot(fig)
    X = pd.get_dummies(X)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X, y)
    fig, ax = plt.subplots()
    ax.scatter(X_pca[:,0], X_pca[:,1], c=y)
    st.pyplot(fig)

    fig, ax = plt.subplots()
    tree = DecisionTreeRegressor().fit(X, y)
    plt.barh(width=tree.feature_importances_, y=X.columns)
    st.pyplot(fig)

    fig, ax = plt.subplots()
    concat.plot('area', 'price(mln)', subplots=True, kind="scatter", ax=ax)
    st.pyplot(fig)

st.sidebar.title('Sidebar')
options = st.sidebar.radio('Страница:', ['Инфо', 'Визуализация', 'Предсказание'])
if options == 'Инфо':
    info()
elif options == 'Визуализация':
    visualize()
elif options == 'Предсказание':
    (ridge_poly, pf, scaler, xgbreg, nn) = load_models()
    st.title('Предсказание')
    X, y = load_test_data()
    with st.form('user_data'):
        area_min, area_max, area_value = int(X[['area']].min()), int(X[['area']].max()), int(X[['area']].mean())
        user_area = st.slider("Площадь", min_value=area_min, max_value=area_max, value=area_value, step=1)

        latitude_min, latitude_max, latitude_value = float(X[['latitude']].min()), float(X[['latitude']].max()), float(X[['latitude']].mean())
        user_latitude = st.slider("latitude", min_value=latitude_min, max_value=latitude_max, value=latitude_value, step=0.001)
        
        longitude_min, longitude_max, longitude_value = float(X[['longitude']].min()), float(X[['longitude']].max()), float(X[['longitude']].mean())
        user_longitude = st.slider("longitude", min_value=longitude_min, max_value=longitude_max, value=longitude_value, step=0.001)

        bedrooms_min, bedrooms_max, bedrooms_value = int(X[['Bedrooms']].min()), int(X[['Bedrooms']].max()), int(X[['Bedrooms']].mean())
        user_bedrooms = st.slider("Bedrooms", min_value=bedrooms_min, max_value=bedrooms_max, value=bedrooms_value, step=1)

        bathrooms_min, bathrooms_max, bathrooms_value = int(X[['Bathrooms']].min()), int(X[['Bathrooms']].max()), int(X[['Bathrooms']].mean())
        user_bathrooms = st.slider("Bathrooms", min_value=bathrooms_min, max_value=bathrooms_max, value=bathrooms_value, step=1)

        balcony_min, balcony_max, balcony_value = int(X[['Balcony']].min()), int(X[['Balcony']].max()), int(X[['Balcony']].mean())
        user_balcony = st.slider("Balcony", min_value=balcony_min, max_value=balcony_max, value=balcony_value, step=1)

        parking_min, parking_max, parking_value = int(X[['parking']].min()), int(X[['parking']].max()), int(X[['parking']].mean())
        user_parking = st.slider("parking", min_value=parking_min, max_value=parking_max, value=parking_value, step=1)

        lift_min, lift_max, lift_value = int(X[['Lift']].min()), int(X[['Lift']].max()), int(X[['Lift']].mean())
        user_lift = st.slider("Lift", min_value=lift_min, max_value=lift_max, value=lift_value, step=1)

        user_status = st.selectbox('Status', X['Status'].unique())
        user_neworold = st.selectbox('neworold', X['neworold'].unique())
        user_furnished_status = st.selectbox('Furnished_status', X['Furnished_status'].unique())
        user_type_of_building = st.selectbox('type_of_building', X['type_of_building'].unique())

        data = pd.DataFrame([[user_area,user_latitude,user_longitude,user_bedrooms,user_bathrooms,user_balcony,user_parking,user_lift,user_status,user_neworold,user_furnished_status,user_type_of_building]],
        columns=X.columns)


        if st.form_submit_button('Предсказать'):
            st.write(data)
            X,y = load_train_data()
            X.loc[-1, 'Status'] = 'Under Construction'     ##!!Kostily alert!!
            X = pd.concat((X, data), axis=0, ignore_index=True)
            X = pd.get_dummies(X)
            data_scaled = scaler.fit_transform(X)
            data_scaled = data_scaled[~np.isnan(data_scaled).any(axis=1)]

            data_scaled_piece = data_scaled[-1:]
            data_poly = pf.transform(data_scaled)
            data_poly_piece = data_poly[-1:]

            st.write('Полиномиальная регрессия')
            poly_pred = ridge_poly.predict(data_poly_piece)
            st.write(poly_pred)

            st.write('Модель XGBoost')
            xgbreg_pred = xgbreg.predict(data_scaled_piece)
            st.write(xgbreg_pred)

            st.write('Нейронная сеть')
            nn_pred = nn.predict(data_scaled_piece)
            st.write(nn_pred)
