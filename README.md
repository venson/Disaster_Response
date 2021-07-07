# Disaster Response Pipeline Project

## 1. Project Motivation

We use disaster data from [Figure Eight](https://www.figure-eight.com/) as the website mentioned, which is [Appen](http://appen.com) at present.
The dataset contains pre-labelled messages when some disaster happens from real-life .
We will use this data to build a model to predict disasters from messages.

Finally we will build a web app to predict disasters from the text we input into the Web app.

With Disaster Response app, we can discover ongoing disasters and emergency situations from twitter or other online massages.
The officials and aids can arrive as soon as possible, which will and must save a lot lives.

1. Clean and prepare the dataset, save the data in SQLlite database file.
2. Extract text features, build and tune model with pipeline and gridsearchcv. Save model to PKL file.
3. Build web app and predict disaster with the model loaded from PKL file.

## 2. Project Features

1. Natural Language Process(NLP) for dataset preparation.
2. Linear classifier model(RidgeClassifier) for fitting and tuning time reduction.
3. CountVectorizer and tfidfTransformer for feature extraction.
3. Pipeline for modeling.
4. GridSearchCV for model tuning.
5. RidgeClassifier and pipeline for modeling.
6. SQLlite and sqlalchemy for data save.
7. Flask for web app.
 


## 3. Installation

+ Archlinux (not recommended to use pip on Archlinux)

  - `sudo pacman -S python-plotly python-pandas python-nltk nltk-data python-sqlalchemy python-flask python-sklearn `

+ Mac 
  - `pip3 install pandas plotly nltk sqlalchemy flask sklearn `

## 4. Usage
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
       1. for linux users: 
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
       2. for mac users: 
        `python3 data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
       1. for linux users: 
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`
       2. for mac users: 
        `python3 models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.

    1. for linux users: 
    `python run.py`
    2. for mac users: 
    `python3 run.py`


