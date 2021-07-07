import json
import joblib
import pandas as pd
import plotly
from flask import Flask, jsonify, render_template, request
from nltk.stem import WordNetLemmatizer
import nltk

#nltk.download(['stopwords', 'punkt', 'wordnet'])
from nltk.tokenize import word_tokenize
from plotly.graph_objs import Bar
#from sklearn.external import extjoblib
from sqlalchemy import create_engine

app = Flask(__name__)


def tokenize(text):
    """
    
    Used as tokenizer for CountVectorizer with word_tokenize() and WordNetlemmatizer() from nltk package.
    The words will be lemmatized with WordNet dict. 

    Parameters:
    ----------
        text : String needed to be tokenized.

    Returns
    ----------
        clean_tokens: a list of words after tokenized and lemmatized.
    """
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    stopwords = set(nltk.corpus.stopwords.words('english'))
    clean_tokens = []
    for tok in tokens:
        if tok not in stopwords:
            clean_tok = lemmatizer.lemmatize(tok).lower().strip()
            clean_tokens.append(clean_tok)

    return clean_tokens


# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('DisasterResponse', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():

    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    # sum of the categoies
    category_sum = df.iloc[:, 4:].sum().transpose().sort_values(
        ascending=False)[:10]

    category_names = list(category_sum.index)

    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    #
    graphs = [
        {
            'data': [Bar(x=genre_names, y=genre_counts)],
            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        ## plot for top 10 categories
        {
            'data': [Bar(x=category_names, y=category_sum)],
            'layout': {
                'title': 'Distribution of TOP 10 Categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category"
                }
            }
        }
    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '')

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file.
    return render_template('go.html',
                           query=query,
                           classification_result=classification_results)


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
