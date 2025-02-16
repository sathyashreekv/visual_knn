import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Streamlit App Title
st.title("🔍 K-Nearest Neighbors Classifier - Pro Edition 🚀")

# Upload CSV File
upload_file = st.file_uploader("📂 Upload your dataset (CSV)", type=['csv'])

if upload_file is not None:
    df = pd.read_csv(upload_file)
    st.write("### 📊 Dataset Preview")
    st.dataframe(df.head())

    # Sidebar for Feature Selection
    st.sidebar.header("⚙️ Model Configuration")
    features = st.sidebar.multiselect("Select Features:", df.columns[:-1])
    target = st.sidebar.selectbox("Select Target column:", df.columns[-1])

    if features and target:
        # Handle missing values
        df = df.dropna()

        # Convert categorical target variable if needed
        if df[target].dtype == 'object':
            df[target] = df[target].astype('category').cat.codes

        X = df[features].values
        y = df[target].values

        # Train-Test Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Feature Scaling
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Select K value
        k = st.sidebar.slider("Select Number of Neighbors (k)", 1, 20, 5)

        if st.sidebar.button("🚀 Train Model"):
            # Train KNN Model
            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(X_train, y_train)
            y_pred = knn.predict(X_test)

            # Metrics
            accuracy = accuracy_score(y_test, y_pred)
            class_report = classification_report(y_test, y_pred, output_dict=True)
            conf_matrix = confusion_matrix(y_test, y_pred)

            # Display results
            st.write(f"### 🎯 Model Accuracy: `{accuracy:.2f}`")
            st.write("### 📑 Classification Report")
            st.dataframe(pd.DataFrame(class_report).T)

            # Confusion Matrix Heatmap
            st.write("### 🔥 Confusion Matrix")
            fig, ax = plt.subplots()
            sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", ax=ax)
            st.pyplot(fig)

            # k-value Tuning Visualization
            st.write("### 📈 Hyperparameter Tuning (Best k)")
            k_range = list(range(1, 21))
            accuracies = [accuracy_score(y_test, KNeighborsClassifier(n_neighbors=i).fit(X_train, y_train).predict(X_test)) for i in k_range]

            fig, ax = plt.subplots()
            ax.plot(k_range, accuracies, marker="o", linestyle="-", color="b")
            ax.set_xlabel("Number of Neighbors (k)")
            ax.set_ylabel("Accuracy")
            ax.set_title("KNN Accuracy vs. K-value")
            st.pyplot(fig)
