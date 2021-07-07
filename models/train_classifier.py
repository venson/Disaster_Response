# import
import sys

import nltk
import pandas as pd

import pickle

#from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sqlalchemy import create_engine
# comment the next line out if run under archlinux
#nltk.download(['punkt', 'wordnet', 'stopwords'])


def load_data(database_filepath):
    """
    Load database from "DisasterResponse" table from sqlite database file.

    Parameters
    -----------
    database_filepath : File path to sqlite database file.

    Returns 
    ----------
    X : Texts from database for modeling, need process before modeling.
    y : category labels for the texts, for modeling, 36 columns total.
    category_names : the category names for y ,used for model evaluation.
    """
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
    """
    Used as tokenizer for CountVectorizer with word_tokenize() and WordNetlemmatizer() from nltk package.
    The words will be lemmatized with WordNet dict. 

    Parameters:
    ----------
        text : String needed to be tokenized.

    Returns
    ----------
        token_list: a list of words after tokenized and lemmatized.




    """

    ## tokenize all the text
    tokens = nltk.word_tokenize(text)
    ## init WordNetLemmatizer
    lemmatizer = nltk.WordNetLemmatizer()

    stopwords = set(nltk.corpus.stopwords.words('english'))
    ## create blank list for return
    token_list = []
    ## lemmatize all the tokenized words that is not a stopword
    for token in tokens:
        if token not in stopwords:
            tk = lemmatizer.lemmatize(token).lower().strip()
            token_list.append(tk)

    return token_list


def build_model():
    """
    build model pipeline for disaster dataset.
    1. CountVectorizer
    2. TfidfTransformer
    3. MultiOutputClassifier with RidgeClassifier
    4. GridSearchCV for tunning(sv = 3, 4 cpu cores)

    Parameters
    ---------
       
    Returns
    ---------
        cv : GridSearchCV model

    """

    #model pipeline for GridSearchCV
    pipeline = Pipeline([('vect', CountVectorizer(tokenizer=tokenize)),
                         ('tfidf', TfidfTransformer()),
                         ('clf', MultiOutputClassifier(RidgeClassifier()))])
    ## param_grid for GridSearchCV
    parameters = {
        'vect__ngram_range': ((1, 1), (1, 2)),
        'vect__max_df': (0.5, 0.75, 1),
        'vect__max_features': (None, 5000, 10000),
        'clf__estimator__alpha': (0.5, 0.75, 1)
    }
    ## GridSearchCV with cv = 3 and cpu 4 cores for shortter time in fiting
    cv = GridSearchCV(pipeline, param_grid=parameters, cv=3, n_jobs=4)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate the model provided.
    Reports each kind of categorys predict accuracy, f1 scores.

    Parameters
    ----------
        model : model need to be evaluated.
        X_test : X for model to predict from
        Y_test : Y for comparison with model predict values
        category_names: names of each columns of Y


    """
    # predict with the model and X data provided
    y_pred = model.predict(X_test)
    # use the prediction to build a dataframe
    y_pred_df = pd.DataFrame(y_pred, columns=category_names)

    # caculate model scores for each category
    for column in category_names:
        print('>>>Report for {} :'.format(column))
        print(
            pd.DataFrame(
                classification_report(Y_test[column],
                                      y_pred_df[column],
                                      zero_division=0,
                                      output_dict=True)))
        print('-------------')


def save_model(model, model_filepath):
    """
    Save the model to a PKL file.

    Parameters
    ------------
        model : model that needed to be saved.
        model_filepath : the path where the model will be save to.

        
    """
    # open a file connection and save the model
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)


def main():
    # run while the execute command has 3 argument values.
    if len(sys.argv) == 3:
        # the second argument value is database filepath
        # the third argument value is model_filepath
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        #load information from database file
        X, Y, category_names = load_data(database_filepath)

        # split train and test data set for model
        X_train, X_test, Y_train, Y_test = train_test_split(X,
                                                            Y,
                                                            test_size=0.2)

        # build model.
        print('Building model...')
        model = build_model()

        # train model with GridSearchCV
        print('Training model...')
        model.fit(X_train, Y_train)

        # evaluating the model accuracy
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        # save model to PKL file
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
