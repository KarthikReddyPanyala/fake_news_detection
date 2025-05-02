# Fake News Detection with SHAP Explainability
project uses ML and SHAP to classify user given articles as **Fake** or **Real**.

## Features
- Logistic regression classifier trained on UCI Fake and Real News dataset.
- TF-IDF vectorization for text preprocessing
- SHAP explaination
- Streamlit interface

## Setup Instructions
### 1. Clone the repo
``` bash
git clone git@github.com:KarthikReddyPanyala/fake_news_detection.git
cd fake_news_detection
```
### 2. Create a virtual Environment
``` bash 
python -m venv venv
source venv/bin/activate
```
### 3. Install Dependencies
``` bash
pip install -r requirements.txt
```
### 3. Dataset
To get the dataset run 
``` bash 
import kagglehub
#Download latest version
path = kagglehub.dataset_download("clmentbisaillon/fake-and-real-news-dataset")
print("Path to dataset files:", path)
```
## 4. run the app
``` bash 
streamlit run fake_news_app.py
```

Notes:
- retrain the model and regenerate the models using Fake_News.ipynb
- Fake_News.ipynb also contains code for data analysis(Data exploration, visualization etc)
