# import
import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    Load data messages and categories form csv file.

    Parameters
    ----------
        messages_filepath : path to messages csv file
        categories_filepath : path to categories csv file

    Returns
    ----------
        df : DataFrame thich have both messages and categories infomations.
    """
    # read files
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    # merge datasets on id
    df = messages.merge(categories, on='id')
    return df


def clean_data(df):
    """
    Clean the dataframe create from load_data()
    separate categories from text into each feature per column

    Parameters
    --------
        df : DataFrame need to be cleaned.

    Returns
    --------
        df :DataFrame after cleaning.

    """
    #split categories
    categories = pd.DataFrame(df.categories.str.split(';', expand=True))
    # use the first row to name the columns
    row = categories.iloc[0, :]
    # use the words except the last 2 characters as columns names
    category_colnames = row.apply(lambda x: x[:-2])
    # rename the columns of `categories`
    categories.columns = category_colnames

    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])

    # drop the original categories column from `df`
    df.drop(['categories'], axis=1, inplace=True)
    # concatenate the processed info together as new 'df'
    df = pd.concat([df, categories], axis=1)
    # remove duplicates
    df = df.drop_duplicates()
    return df


def save_data(df, database_filename):
    """
    Save Cleaned dataframe in sqlite database file with table "DisasterResponse"
    
    Parameters
    ----------
        df: dataframe need to be saved
        database_filename : path to sqlite database file

    """
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('DisasterResponse', engine, index=False)


def main():

    ## run with the command have 4 argument values
    if len(sys.argv) == 4:

        ## get argument values to variables
        ## 1st argument value : path of messages csv file to be loaded
        ## 2nd argument value : path of categories csv file to be loaded
        ## 3rd argument value : path of sqlite database file to be saved to
        messages_filepath, categories_filepath, database_filepath = sys.argv[
            1:]

        #load csv file to dataframe
        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'.format(
            messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        # clean the dataset
        print('Cleaning data...')
        df = clean_data(df)

        # save the cleaned dataframe to sqlite database file
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
