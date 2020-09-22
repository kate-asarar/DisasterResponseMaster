import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql("SELECT * FROM DisasterResponse", engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    # define function for counting messages by category
    def get_category_count(df = df): 
        category_counts = {}
        for column in df.columns[4:]:
            try: 
                category_counts[column] =df.groupby(column).count()['message'][1]
            except:
                category_counts[column] = 0
        return category_counts
    # extract first visual 
    category_counts = get_category_count()
    category_names = list(category_counts.keys())
    

    def get_category_count_per_genre():
        dict_categories = {}
        for genre in genres: 
            df_genre = df[df['genre'] == genre]
            dict_categories[genre] = get_category_count(df = df)
        return pd.DataFrame(dict_categories)
    genres = df['genre'].unique()
    df_categories = get_category_count_per_genre()

    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x= genre_names,
                    y= genre_counts
                )
            ],

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
         {
            'data': [
                Bar(
                    x= category_names,
                    y= list(category_counts.values())
                )
            ],

            'layout': {
                'title': 'Distribution of Message categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category"
                }
            }
        }, 
         {
            'data': [
                Bar (
                     x = list(df_categories.index), 
                     y = list(df_categories[genres[0]]),
                     name = genres[0]
                       ), 
                Bar (
                     x = list(df_categories.index), 
                     y = list(df_categories[genres[1]]),
                     name = genres[1]
                       ), 
                Bar (
                     x = list(df_categories.index), 
                     y = list(df_categories[genres[2]]),
                     name = genres[2]
                       )
            ],

            'layout': {
                'title': 'Distribution of Message categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category"
                }, 
                'barmode': 'stack'
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
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()