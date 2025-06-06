import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import argparse


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

    posts_file_path = f'data/{dataset_name}/csv/Posts.csv'
    data_path = f'data/{dataset_name}/csv'

    if not os.path.exists(data_path):
        os.makedirs(data_path)

    try:

        dataset_path = f'user_pre_training/data/{dataset_name}'
        if not os.path.exists(dataset_path):
            os.makedirs(dataset_path)

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
        
        df['QId'] = le2.fit_transform(df['QId'])
        
        print("Performing train test split...\n")
        # Splitting the DataFrame

        df['AnsweredDate'] = pd.to_datetime(df['AnsweredDate'])

        # Sort by AnsweredDate to get temporal order
        question_order = df.groupby('QId')['AnsweredDate'].min().sort_values().index.tolist()

        # Total number of unique questions
        total_questions = len(question_order)

        # Calculate splits
        pretrain_q_count = int(total_questions * 0.6)
        rem_count = int(total_questions * 0.4)

        # Get QIds for each split
        pretrain_qids = question_order[:pretrain_q_count]
        rem_qids = question_order[pretrain_q_count:]

        # Filter data for each split
        pretrain_df = df[df['QId'].isin(pretrain_qids)]
        rem_df = df[df['QId'].isin(rem_qids)]

        user_ids = list(pretrain_df['UserId'])
        tags = list(pretrain_df['Qtag'])

        pretrain_df = pretrain_df.loc[pretrain_df['Top_user'].isin(user_ids)]

        rem_df = rem_df.loc[rem_df['UserId'].isin(user_ids)]
        rem_df = rem_df.loc[rem_df['Top_user'].isin(user_ids)]
        rem_df = rem_df.loc[rem_df['Qtag'].isin(tags)]

        pretrain_df['UserId'] = le.fit_transform(pretrain_df['UserId'])
        pretrain_df['Top_user'] = le.transform(pretrain_df['Top_user'])
        pretrain_df['Qtag'] = le1.fit_transform(pretrain_df['Qtag'])

        rem_df['UserId'] = le.transform(rem_df['UserId'])
        rem_df['Top_user'] = le.transform(rem_df['Top_user'])
        rem_df['Qtag'] = le1.transform(rem_df['Qtag'])

        rem_df.to_csv(f'{dataset_path}/answers_data.csv',header = False,index=False)
        

        train_data, tuning_data, test_data = split_user_data(pretrain_df)

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

        print("===================================PRE TRAIN====================================================")
        print("No. of. Users",pretrain_df['UserId'].max()+1)
        print()
        print("No. of. Tag combinations",pretrain_df['Qtag'].max()+1)
        print()
        print("No of Questions : ",pretrain_df['QId'].nunique())
        print("=======================================================================================")

        print("===================================REM====================================================")
        print("No. of. Users",rem_df['UserId'].max()+1)
        print()
        print("No. of. Tag combinations",rem_df['Qtag'].max()+1)
        print()
        print("No of Questions : ",rem_df['QId'].nunique())
        print("=======================================================================================")

        print("Preprocessing completed and all necessary files are created\n")

    except Exception as e:
        print(e)
        print("!!! Error  occured while processing the data unable to proceed !!!\n")