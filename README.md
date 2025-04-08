# Expert_recommendation

1.Download datasets from :

	https://archive.org/download/stackexchange

2.Extract 7z file using 7-zip file manager.

3.Convert the xml files to csv files :

Follow the steps in readme of :

	https://github.com/SkobelevIgor/stackexchange-xml-converter 
 (skip the extract part as we already have done it) to convert xml files to CSV files.

Copy these csv files into data folder and create a folder with your dataset name and inside that create a folder with name csv and paste User.csv and Posts.csv files.

4.Preprocess your data using preprocess code :
	
 	Ex : python3 preprocess.py --dataset_name ask_ubuntu

You will see a result like this after successful preprocessing
![image](https://github.com/user-attachments/assets/4bf07723-081f-426d-8eaf-e35ff8b54432)


Then add the no of user in user_dict and no of tag combinations in tag_dict in following format

NOTE: every key should be the same as the folder name in which the dataset is present.


In the main file increase or decrease dimensions of the embeddings based on no of users. In opt.d_model attribute.


5.Pretrain the embedding model for better expert embeddings
Open a terminal in Expert_Recommendation folder and Use this command to pre train : 
python3 user_pre_training/Main.py
Now you should find a saved model in the results folder.

6.Now train the scoring function to score the experts 
Open a terminal in Expert_Recommendation folder and Use this command to train scoring function: 
python3 Expert_pred/Main.py
