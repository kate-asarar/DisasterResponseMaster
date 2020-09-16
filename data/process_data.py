import sys
import sqlite3
import pandas as pd
import re
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    ''' Load data into pandas dataFrame
    parameters:
        messages_filepath: The path to the messages csv file
        categories_filepath: The path to the message categories csv file

    returns:
        df: DataFrame with both mesages and their categories
    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(categories, messages, left_on='id', right_on='id')
    return df


def clean_data(df):
    ''' Expand categories column into categorical columns
    and remove duplicates

    parameters:
        df: DataFrame with messages and categories column

    returns:
        df: DataFrame
    '''
    # expand the categories column into categorical binary class matrix
    categories = pd.Series(df['categories']).str.split(pat=';', expand=True)

    # get column names from the categories in the first row
    category_colnames = categories.iloc[0].apply(lambda x: re.sub(r'(-[0-9])', '', x))

    # change categories column names
    categories.columns = category_colnames
    # remove category names from values in columns and parse the values into int
    for column in categories:
        categories[column] = categories[column].apply(lambda x: re.sub(r'([^0-9])', '', x))
        categories[column] = categories[column].apply(lambda x: int(x))

    # replace categories in df with expanded categories
    df.drop(['categories'], axis=1, inplace=True)
    df = pd.concat([df, categories], sort=False)

    # remove duplicates
    df = df.drop_duplicates()

    return df


def save_data(df, database_filename):
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql(database_filename, engine, index=False)
    pass

def main():
    if len(sys.argv) == 4:
        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        engine = create_engine('sqlite:///{}'.format(database_filepath))
        df = df.dropna(axis=0)
        engine.execute("DROP TABLE IF EXISTS DisasterResponse")
        df.to_sql('DisasterResponse', engine, index=False)
        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories ' \
              'datasets as the first and second argument respectively, as ' \
              'well as the filepath of the database to save the cleaned data ' \
              'to as the third argument. \n\nExample: python process_data.py ' \
              'disaster_messages.csv disaster_categories.csv ' \
              'DisasterResponse.db')


if __name__ == '__main__':
    main()