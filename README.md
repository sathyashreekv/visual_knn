# visual_knn
# K-Nearest Neighbors (KNN) Classifier using Streamlit

## 📌 Introduction

This is a simple **K-Nearest Neighbors (KNN) Classifier** built using **Streamlit**. The app allows users to upload a dataset, select features and a target variable, train the KNN model, and view the classification accuracy along with visualizations.

## 🚀 Features

- 📂 **Upload CSV** – Users can upload any dataset.
- 🔧 **Feature Selection** – Choose specific features and target.
- 🎛 **Adjust ****`k`**** (Number of Neighbors)** – Interactive slider to modify `k`.
- 🎯 **Train Model** – Click a button to train KNN on the selected dataset.
- 📊 **View Accuracy** – Displays the model's classification accuracy.


## 🛠 Installation

To run this Streamlit app, install the required dependencies:

```bash
pip install streamlit pandas numpy matplotlib seaborn scikit-learn
```

## ▶️ How to Run

Save the script as `knn.py` and run the following command:

```bash
streamlit run knn.py
```

Then open the **local Streamlit URL** in your browser.

## 📜 How It Works

1. Upload a **CSV dataset**.
2. Select **features** and the **target column**.
3. Choose a value for **k (number of neighbors)** using the slider.
4. Click **"Train KNN Model 🚀"** to train the model.
5. View **model accuracy**.

## 🖥 Example Dataset Format

Ensure your dataset has a structure like this:

| Feature 1 | Feature 2 | Target |
| --------- | --------- | ------ |
| 5.1       | 3.5       | 0      |
| 4.9       | 3.0       | 1      |
| 6.2       | 2.8       | 1      |
| 5.8       | 2.7       | 0      |

## 📌 Code Snippet (

```python
from sklearn.metrics import accuracy_score

knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
st.write(f"### Model Accuracy: {accuracy:.2f}")
```


## 🤝 Contributing

Feel free to fork the repository and submit pull requests! 🚀

## 📜 License

This project is open-source and available under the **MIT License**.

---

Enjoy using the KNN Streamlit App! 🎉

