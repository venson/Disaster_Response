# import
import sys

import nltk
import pandas as pd

nltk.download(['punkt', 'wordnet'])
import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sqlalchemy import create_engine


def load_data(database_filepath):
    # create engine for database file
    engine = create_engine('sqlite:///' + database_filepath)
    # read table
    df = pd.read_sql_table('DisasterResponse', con=engine)
    # use 'message' column as X
    X = df['message']
    # use fifth column and the right as y
    y = df.iloc[:, 4:]
    category_names = y.columns
    return X, y, category_names


def tokenize(text):
    tokens = nltk.word_tokenize(text)
    lemmatizer = nltk.WordNetLemmatizer()

    token_list = []
    for token in tokens:
        tk = lemmatizer.lemmatize(token).lower().strip()
        token_list.append(tk)

    return token_list


def build_model():
    pipeline = Pipeline([('vect', CountVectorizer(tokenizer=tokenize)),
                         ('tfidf', TfidfTransformer()),
                         ('clf', MultiOutputClassifier(RidgeClassifier()))])
    parameters = {
        'vect__ngram_range': ((1, 1), (1, 2)),
        'vect__max_df': (0.5, 0.75, 1),
        'vect__max_features': (None, 5000, 10000),
        'clf__estimator__alpha': (0.5, 0.75, 1)
    }
    cv = GridSearchCV(pipeline, param_grid=parameters, cv=3, n_jobs=4)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    y_pred = model.predict(X_test)
    y_pred_df = pd.DataFrame(y_pred, columns=category_names)
    for column in category_names:
        print('Report for {} :'.format(column))
        print(
            classification_report(Y_test[column],
                                  y_pred_df[column],
                                  zero_division=0))
        print('--------------------------------')


def save_model(model, model_filepath):
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X,
                                                            Y,
                                                            test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
