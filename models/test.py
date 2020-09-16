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

def main():
    database_filepath = sys.argv[1:]
    load_data(database_filepath)

if __name__ == '__main__':
    main()