# AutoML Platform

A Streamlit-based AutoML application that enables users to upload datasets, automatically analyze them, train multiple machine learning models, compare model performance, and generate predictions without writing code.

---

## Features

### Data Upload
- Upload datasets in CSV format
- Preview uploaded data
- Validate uploaded files

### Exploratory Data Analysis (EDA)
- Dataset overview
- Shape information
- Missing values analysis
- Numerical and categorical feature inspection
- Statistical summaries

### Automatic Preprocessing
- Missing value handling
- Categorical encoding
- Feature scaling
- Automated preprocessing pipeline

### Machine Learning
- Automatic task detection
  - Classification
  - Regression
- User model selection
- Training multiple models simultaneously
- Model comparison dashboard

### Prediction
- Upload new data
- Generate predictions using trained models
- Download prediction results as CSV

---

## Project Structure

```text
MyAutoML/
│
├── config/
│   └── config.py
│
├── pages/
│   ├── 1_Upload.py
│   ├── 2_EDA.py
│   ├── 3_Training.py
│   └── 4_Prediction.py
│
├── src/
│   ├── data/
│   │   ├── loader.py
│   │   ├── split.py
│   │   ├── preprocessing.py
│   │   └── EDA.py
│   │
│   ├── models/
│   └── utils/
│
├── main.py
├── requirements.txt
└── README.md
```

---

## Technologies Used

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-Learn
- Matplotlib
- Seaborn

---

## Installation

Clone the repository:

```bash
git clone https://github.com/Alaaeldin013/MyAutoML.git
cd MyAutoML
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Application

```bash
streamlit run main.py
```

---

## Workflow

1. Upload a CSV dataset.
2. Select the target column.
3. Explore the dataset through EDA.
4. Choose machine learning models.
5. Train and compare models.
6. Select a model for prediction.
7. Upload new data and generate predictions.
8. Download prediction results.

---

## Future Improvements

- Hyperparameter tuning
- Feature importance visualization
- Experiment tracking with MLflow
- Model persistence and loading
- Deployment to Streamlit Cloud
- Advanced AutoML optimization

---

## Author

Alaaeldin Mohamed
