import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import argparse

# Function for getting answerer ID (this is later used to get user id of accepted answerers)
def get_user_id(post_id):
    
    try:
        x = answerposts[answerposts['Id']==post_id]['OwnerUserId'].values[0]
    except:
        print(post_id)
    return x

# Function to split the data.
def split_user_data(df, test_size=0.2, val_size=0.1):
    train_list = []
    val_list = []
    test_list = []
    
    for user_id, group in df.groupby('UserId'):
        
        group = group.sort_values('AnsweredDate')[['UserId','Qtag']]
        
        if(group.shape[0]<3):
            train_list.append(group)
            continue
        
        train_and_val, test = train_test_split(group, test_size=test_size, shuffle=True, random_state=42)
        train, val = train_test_split(train_and_val, test_size=val_size/(1-test_size), shuffle=True, random_state=42)
        
        train_list.append(train_and_val)
        val_list.append(val)
        test_list.append(test)
    
    train_df = pd.concat(train_list).reset_index(drop=True)
    val_df = pd.concat(val_list).reset_index(drop=True)
    test_df = pd.concat(test_list).reset_index(drop=True)
    
    return train_df, val_df,test_df

def make_txt_file(df, output_file):
    # Open output file for writing
    with open(output_file, 'w') as out_file:
        # Iterate over the DataFrame rows
        for index, row in df.iterrows():
            # Convert each row into a tab-separated string
            line = '\t'.join(map(str, row.values))  # Convert all values to strings

            # Write the line to the output file
            out_file.write(line + '\n')

    print(f'Conversion complete. TXT file saved as {output_file}')

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Pass dataset name")
    parser.add_argument("--dataset_name", required=True, help="name of dataset")
    parser.add_argument("--min_answers", required=True, type=int, help="minimum no of questions to be answered by an expert")

    # Parse the arguments
    args = parser.parse_args()

    # Access the arguments
    dataset_name = args.dataset_name
    min_answers = args.min_answers

    user_file_path = f'data/{dataset_name}/csv/Users.csv'
    posts_file_path = f'data/{dataset_name}/csv/Posts.csv'
    data_path = f'data/{dataset_name}/csv'

    if not os.path.exists(data_path):
        os.makedirs(data_path)

    try :
        print("\nReading posts....\n")

        posts = pd.read_csv(posts_file_path)

        posts['Id'] = posts['Id'].astype(int)
        posts['Score'] = posts['Score'].astype(int)
        posts = posts[posts['OwnerUserId'].notna()]
        posts['OwnerUserId'] = posts['OwnerUserId'].astype(int)

        print("\nposts Read succesfully\n")
    except:
        print("!!! Unable to read posts file !!!\n")

    try :
        print("seperating answer posts from Posts.csv\n")
        
        answerposts = posts[(posts['PostTypeId']==2)]
        answerposts['OwnerUserId'] = answerposts['OwnerUserId'].astype(int)
        answerposts['ParentId'] = answerposts['ParentId'].astype(int)
        answerposts['CreationDate']  = pd.to_datetime(answerposts['CreationDate'] )

        answerposts = answerposts.reset_index(drop=True)

        print("seperation succesful\n")
    except : 
        print("!!! Unable to seperate answer posts !!!\n")

    try : 

        print("seperating question posts from Posts.csv\n")

        questionposts = posts[(posts['PostTypeId']==1)]
        questionposts = questionposts[questionposts['AcceptedAnswerId'].notna()]
        questionposts['AcceptedAnswerId'] = questionposts['AcceptedAnswerId'].astype(int)
        questionposts['OwnerUserId'] = questionposts['OwnerUserId'].astype(int)

        print("seperation succesful\n")
    except : 
        print("!!! Unable to seperate question posts !!!\n")
    
    try : 
        
        print("Preprocessing the data...\n")

        #selecting answers only for the questions in questionposts
        questionids  = list(set(list(questionposts['Id']))) 
        answerposts = answerposts.loc[answerposts['ParentId'].isin(questionids)]

        #selecting only questions that have accepted answer 
        answerids = list(set(list(answerposts['Id'])))
        questionposts = questionposts.loc[questionposts['AcceptedAnswerId'].isin(answerids)]

        questionposts['Accepted_answerer_Id'] = questionposts['AcceptedAnswerId'].apply(get_user_id)
        questionposts = questionposts.reset_index(drop=True)

        # Initialize an empty list to store dictionaries
        data = []

        # Loop through each row in the answers DataFrame
        for index, row in answerposts.iterrows():
            # Create an empty dictionary for each row
            row_dict = {}
            
            # Fill in the dictionary with data from each column
            row_dict['UserId'] = row['OwnerUserId']
            
            qrow = questionposts[questionposts['Id'] == row['ParentId']]
            
            # If there is no matching question, skip this row
            if qrow.empty :
                continue
            
            #preprocessing the tags
            tag = qrow['Tags'].values[0]
            tag = tag.split("|")
            tag = tag[1:-1]
            tag.sort()
            qtag = "|".join(tag)
            
            row_dict['Qtag'] = qtag
            
            row_dict['AnsweredDate'] = row['CreationDate']
            
            row_dict['Top_user'] = int(qrow['Accepted_answerer_Id'])
            
            row_dict['QId'] = int(qrow['Id'])

            #Append the dictionary to the list
            #print(row_dict)
            #break 
            data.append(row_dict)        

        print("Creating dataframe...\n")
        
        dataset_path = f'user_pre_training/data/{dataset_name}'
        if not os.path.exists(dataset_path):
            os.makedirs(dataset_path)

        data = pd.DataFrame(data)
        data.to_csv(f'{dataset_path}/temporary_data.csv',index=False)

        print(f"dataframe saved at {dataset_path}\n")

        data = pd.read_csv(f'{dataset_path}/temporary_data.csv')

        #If a user has not given any accepted answer, that user shouldnt be considered as expert as of now

        valid_users = list(set(list(data['Top_user'])))
        data = data.loc[data['UserId'].isin(valid_users)]

        user_counts = data['UserId'].value_counts()

        users_to_keep = user_counts[user_counts>min_answers].index

        df = data.loc[data['UserId'].isin(users_to_keep)]
        df = df.loc[df['Top_user'].isin(users_to_keep)]

        le = LabelEncoder()
        le1 = LabelEncoder()
        le2 = LabelEncoder()

        df['UserId'] = le.fit_transform(df['UserId'])
        df['Top_user'] = le.transform(df['Top_user'])
        df['Qtag'] = le1.fit_transform(df['Qtag'])
        df['QId'] = le2.fit_transform(df['QId'])

        df.to_csv(f'{dataset_path}/answers_data.csv',header = False,index=False)

        print("Performing train test split...\n")
        # Splitting the DataFrame
        train_data, tuning_data, test_data = split_user_data(df)

        train_data = train_data.groupby(['UserId', 'Qtag']).size().reset_index(name='count')

        tuning_data = tuning_data.groupby(['UserId', 'Qtag']).size().reset_index(name='count')

        test_data = test_data.groupby(['UserId', 'Qtag']).size().reset_index(name='count')
        
        print("Converting dataframes into text files...\n")        

        output_file = f'{dataset_path}/{dataset_name}_train.txt' 
        make_txt_file(train_data,output_file)

        output_file = f'{dataset_path}/{dataset_name}_tune.txt'  
        make_txt_file(tuning_data,output_file)

        output_file = f'{dataset_path}/{dataset_name}_test.txt'  
        make_txt_file(test_data,output_file)

        print("=======================================================================================")
        print("No. of. Users",df['UserId'].max()+1)
        print()
        print("No. of. Tag combinations",df['Qtag'].max()+1)
        print("=======================================================================================")

        print("Preprocessing completed and all necessary files are created\n")

    except:
        print("!!! Error  occured while processing the data unable to proceed !!!\n")


