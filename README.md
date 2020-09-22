# Disaster Response Pipeline Project
This project consits of three parts: 
1) a NLP ETL script which loads in message data as well as their corresponding categories, clean the data and then saves everything in a new dataframe. 
2) a machine learning message classification script which fits a randomtree model to the data and evaluates the model. The model can be obtipized using grid search or not. 
3) a website app that desplays the results of the classification. 

### The reposotory contains the following data files in the data folder: 
1) disaster_messages.csv -- contains the messages that are to be classified 
2) disaster_categories.csv -- contains the correct classification labels for the messages
  
### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
