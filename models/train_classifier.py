import sys
import sqlite3
import pandas as pd
import re
import sqlalchemy as db
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import nltk
nltk.download(['punkt', 'wordnet'])
import numpy as np

def load_data(database_filepath):
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql("SELECT * FROM DisasterResponse", engine)
    X = df['message']
    Y= df[df.columns[4:]]
    category_names = Y.columns
    print('Successfully loaded data!')
    return X, Y, category_names


def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    return clean_tokens

def create_model_pipeline(optimize):
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    if optimize:
        parameters = {
        'vect__ngram_range': [(1,1), (1,3)],
        'tfidf__norm': ('l1', 'l2', None),
        }

        cv = GridSearchCV(pipeline, param_grid=parameters, scoring='f1')
        return cv
    else:
        return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    y_pred = model.predict(X_test)
    confusion_mat = confusion_matrix(Y_test, y_pred, labels=category_names)
    accuracy = (y_pred == Y_test).mean()
    return accuracy, confusion_mat


def save_model(model, model_filepath):
    pass


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = create_model_pipeline(optimize=True)
        
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