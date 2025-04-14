import sys
import torch
import pandas as pd
import Constants as C

from tqdm import tqdm
from Src.Models import Embed_Model
from torch.utils.data import Dataset
from preprocess.Dataset import Dataset as dataset
from sklearn.model_selection import train_test_split


def calculate_embeds(opt,embed_path):
    """ Epoch operation in evaluation phase. """

    model = Embed_Model(
        num_types=C.TAG_NUMBER,
        d_model=opt.d_model,
        n_layers=opt.n_layers,
        n_head=opt.n_head,
        dropout=opt.dropout,
        device=opt.device
    )
    
    model.load_state_dict(torch.load(embed_path+'.pth'),strict=False)
    model = model.cuda()    

    ds = dataset()
    user_dl = ds.get_user_dl(opt.batch_size)
    user_valid_dl = ds.get_user_valid_dl(opt.batch_size)
    data = (user_valid_dl, user_dl)

    model.eval()

    user_embeds = []
    with torch.no_grad():
        for batch in tqdm(user_dl, mininterval=2,
                          desc='  - (Generating embeddings) ', leave=False):
            """ prepare test data """
            event_type, event_time, test_label = map(lambda x: x.to(opt.device), batch)

            """ forward """
            prediction, users_embeddings = model(event_type)  # X = (UY+Z) ^ T

            user_embeds.append(prediction)

    final_embeds = torch.cat(user_embeds,dim=0)
    final_q_embeds = model.state_dict()['event_emb.weight']
    
    torch.save(final_embeds,embed_path+'_user_embeds.pt')
    torch.save(final_q_embeds,embed_path+'_ques_embeds.pt')

    print("============================================================")
    print("final_user_embeds : ",final_embeds.size())
    print("final_ques_embeds : ",final_q_embeds.size())
    print("\nEmbeddings for users and questions saved succesfully.......")
    print("============================================================")


def give_data(data_path,min_users=3):
    
    df = pd.read_csv(f'{data_path}/answers_data.csv',header=None,names=['UserId','Qtag_Id','Timestamp','Top_userId','QId'])

    df = df.drop(columns=['Timestamp'])

    df['Qtag_Id'] = df['Qtag_Id']+1

    ques_counts = df['QId'].value_counts()
    ques_to_keep = ques_counts[ques_counts>=min_users].index

    df = df.loc[df['QId'].isin(ques_to_keep)]

    print(f"\n{len(ques_to_keep)} valid questions were found")

    #for this data i found that 23000 questions have atleast 2 users answered them

    data = df.groupby('QId').agg({
        'UserId': list,  # Collect answerer ids into a list
        'Top_userId': 'first',  # Top user is the same for all rows with the same question_id
        'Qtag_Id':'first'
        }).reset_index()

    #Remove
    # length_counts = {}

    # for user_list in data['UserId']:
    #     length = len(user_list)  
    #     if length in length_counts:
    #         length_counts[length] += 1
    #     else:
    #         length_counts[length] = 1

    # for key, value in sorted(length_counts.items()):
    #     print(f"Length: {key}, Count: {value}")
    
    #Remove
    temp =  data['UserId'].apply(len)

    train, test = train_test_split(data,test_size = 0.2)

    print("\n====================================================================================")
    
    print("Maximum length of 'UserId' list:", temp.max())

    print(f"No of questions in train : {len(train)}")
    print(f"No of questions in test : {len(test)}\n")

    print("Data set 5 samples :\n")
    print(data.head())
    print("\n====================================================================================")

    return train,test



class RankingDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        
        question_id = self.data.iloc[idx]['QId']
        top_user_id = self.data.iloc[idx]['Top_userId']
        answerer_ids = self.data.iloc[idx]['UserId']
        q_tag_id = self.data.iloc[idx]['Qtag_Id']

        return {
            'UserId': torch.tensor(answerer_ids),
            'QId': torch.tensor(question_id),
            'Top_userId': torch.tensor(top_user_id),
            'Qtag_Id' : torch.tensor(q_tag_id)
        }

def get_embeddings(batch, user_embeddings, question_embeddings):

    user_batch_ids = batch['UserId'].clone().detach()
    top_user_ids = batch['Top_userId'].clone().detach()
    q_tag_id = batch['Qtag_Id'].clone().detach()

    # Get embeddings by indexing
    user_batch_embeddings = user_embeddings[user_batch_ids]
    top_user_embeddings = user_embeddings[top_user_ids]
    q_tag_embeddings = question_embeddings[q_tag_id]
    
    return user_batch_embeddings, top_user_embeddings, q_tag_embeddings

def performance_metrics(aid_list, score_list, accid, k):
        """
        Performance metric evaluation

        Args:
            aid_list  -  the list of aid in this batch
            score_list  -  the list of score of ranking
            accid  -  the ground truth
            k  -  precision at K
        """

        if(aid_list.size()[1]==1):
            return int(aid_list[0] == accid), 1, 1

        aid_list = aid_list.squeeze()
        score_list = score_list.squeeze()

        if len(aid_list) != len(score_list):
            print("aid_list and score_list not equal length.",
                  file=sys.stderr)
            sys.exit()

        id_score_pair = list(zip(aid_list, score_list))
        id_score_pair.sort(key=lambda x: x[1], reverse=True)

        for ind, (aid, score) in enumerate(id_score_pair):
            if aid == accid:
                if ind == 0:
                    return 1/(ind+1), int(ind < k), 1
                else:
                    return 1/(ind+1), int(ind < k), 0

            

if __name__ == "__main__":
    data_path=f'/u/student/2023/cs23mtech11032/Punith/Expert_recommendation/user_pre_training/data/{C.DATASET}'
    
    give_data(data_path)