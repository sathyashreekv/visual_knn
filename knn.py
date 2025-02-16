import streamlit as st
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


st.title("K-Nearest Neighbhors Classifer 🔍")

upload_file=st.file_uploader("Upload csv file",type=['csv'])

if upload_file is not None:
    df=pd.read_csv(upload_file)
    st.write("preview of Dataset")
    st.write(df.head())

    features=st.multiselect("Select Features :",df.columns[:-1])
    target=st.selectbox("Select Target column :",df.columns[-1])

    if features and target:
        x=df[features].values
        y=df[target].values

        x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

        scaler=StandardScaler()
        x_train=scaler.fit_transform(x_train)
        x_test=scaler.transform(x_test)

        k=st.slider("Select Number of Neighbors (k)", 1,20,5)
        if st.button("Train KNN Model 🚀"):
            knn=KNeighborsClassifier(n_neighbors=k)
            knn.fit(x_train,y_train)
            y_pred=knn.predict(x_test)

            accuracy=accuracy_score(y_test,y_pred)
            st.write(f"Model Accuracy :{accuracy:.2f}")
