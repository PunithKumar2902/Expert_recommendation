import os
import torch
import argparse
import Constants as C

from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence

from utils import *

from torch.utils.data import DataLoader
from Src.Models import Ranking_model

import random
import numpy as np

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # Disable CuDNN optimizations for reproducibility



def collate_fn(batch):

    user_tensors = [data['UserId'].clone().detach() for data in batch]
    user_tensors_padded = pad_sequence(user_tensors, batch_first=True, padding_value=0)

    top_users = torch.tensor([data['Top_userId'] for data in batch])
    qtag_ids = torch.tensor([data['Qtag_Id'] for data in batch])

    return {'UserId': user_tensors_padded, 'Top_userId': top_users, 'Qtag_Id': qtag_ids}


def train(model, dl, user_embeds, ques_embeds, opt):

    parameters = [{'params': model.parameters(), 'lr': opt.lr},]
    optimizer = torch.optim.Adam(parameters)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, 10, gamma=0.5)
   
    model.train()
    for epoch in range(opt.epoch):

        print(f"\n[Epoch : {epoch+1}]")
        with open(f"outputs/{C.DATASET}_output.txt", "a") as file:
            file.write(f"\n[Epoch : {epoch+1}]")

        tot_loss=0
        for batch in tqdm(dl, mininterval=2, desc='  - (Training)   ', leave=False):

            optimizer.zero_grad()
            users, top_users, questions = get_embeddings(batch, user_embeds, ques_embeds)
            neg_samples,neg_indices = sample_negative_users(user_embeds, batch['UserId'],50)

            loss = model(users, top_users, questions,batch['UserId'], neg_samples)

            tot_loss+=loss

            loss.backward()

            optimizer.step()
            # scheduler.step()

        print(f"Epoch {epoch+1} completed with total loss {tot_loss}")
        print(f"Epoch {epoch+1} completed with Avg loss {tot_loss/len(dl)}\n")
        with open(f"outputs/{C.DATASET}_output.txt", "a") as file:
            file.write(f"\nEpoch {epoch+1} completed with total loss {tot_loss}\n")
            file.write(f"Epoch {epoch+1} completed with Avg loss {tot_loss/len(dl)}\n")

def test(model, dl, user_embeds, ques_embeds):

    k=5
    MRR, hit_K, prec_1 = 0, 0, 0

    print("===============================================================")
    print("Test ques: ",len(dl))
    print("===============================================================")

    model.eval()
    with torch.no_grad():
        
        for batch in tqdm(dl, mininterval = 2, desc='  - (Testing) ', leave=False):
            
            users, top_users, questions = get_embeddings(batch, user_embeds, ques_embeds)

            if len(list(users.size()))>2:
                users = users.squeeze(0)
            
            # neg_samples,neg_indices = sample_negative_users(user_embeds, batch['UserId'],50)

            # neg_samples = neg_samples.squeeze(0)

            # users = torch.cat([users,neg_samples],dim=0)

            scores = model.test(users,questions)

            user_batch_ids = batch['UserId'].clone().detach().view(-1)
            # neg_indices = neg_indices.view(-1)

            # user_batch_ids = torch.cat([user_batch_ids,neg_indices],dim =0)

            top_user_ids = batch['Top_userId'].clone().detach()

            R_R, hit, prec = performance_metrics(user_batch_ids,scores,top_user_ids,k) 

            MRR += R_R
            hit_K += hit
            prec_1 += prec

        print("===============================================================")
        print("MRR : ",MRR)
        print("hit_K : ",hit_K)
        print("prec_1 : ",prec_1) 
        print("===============================================================")

        print("===============================================================")
        print("MRR : ",MRR/len(dl))
        print("hit_K : ",hit_K/len(dl))
        print("prec_1 : ",prec_1/len(dl)) 
        print("===============================================================")

        with open(f"outputs/{C.DATASET}_output.txt", "a") as file:
            file.write("===============================================================\n")
            file.write(f"MRR : {MRR}\n")
            file.write(f"hit_K : {hit_K}\n")
            file.write(f"prec_1 : {prec_1}\n")
            file.write("===============================================================\n")

            file.write("===============================================================\n")
            file.write(f"MRR : {MRR/len(dl)}\n")
            file.write(f"hit_K : {hit_K/len(dl)}\n")
            file.write(f"prec_1 : {prec_1/len(dl)}\n")
            file.write("===============================================================\n")


if __name__ == '__main__':
    set_seed(42)  
    parser = argparse.ArgumentParser()
    opt = parser.parse_args()
    opt.device = torch.device('cuda')

    opt.epoch = 5
    opt.n_layers = 1  

    opt.batch_size = 8
    
    opt.dropout = 0.6
    opt.smooth = 0.06
    opt.n_head = 1
    opt.d_model = C.EMB_size

    opt.lr = 0.0001

    embed_path = f"results/{C.DATASET}/{C.DATASET}"
    data_path=f'user_pre_training/data/{C.DATASET}'

    if not (os.path.exists(embed_path+'_user_embeds.pt') and os.path.exists(embed_path+'_ques_embeds.pt')):

        print("\n==========================================================")
        print("Embedding files not found, generating .......")
        calculate_embeds(opt,embed_path)
        print("Embeddings generated succesfully....")
        print("============================================================\n")

    user_embeds, ques_embeds = torch.load(embed_path+'_user_embeds.pt'),torch.load(embed_path+'_ques_embeds.pt')
    
    print("============================================================")
    user_embeds = user_embeds.to(opt.device)
    ques_embeds = ques_embeds.to(opt.device)
    print("final_user_embeds : ",user_embeds.size())
    print("final_ques_embeds : ",ques_embeds.size())
    print("\nEmbeddings loaded succesfully .....")
    print("============================================================")

    # train_data, test_data = give_data(data_path)
    train_data, test_data = give_data(data_path,1)

    train_data = RankingDataset(train_data)
    test_data = RankingDataset(test_data)

    # train_dl = DataLoader(train_data, batch_size=8, shuffle=True,collate_fn = collate_fn)
    train_dl = DataLoader(train_data, batch_size=8, shuffle=False, collate_fn = collate_fn)
    # test_dl = DataLoader(test_data, batch_size=1, shuffle=True)
    test_dl = DataLoader(test_data, batch_size=1, shuffle=False)

    model = Ranking_model(opt.d_model,C.KNL_size,1)

    model = model.to(opt.device)

    train(model, train_dl, user_embeds, ques_embeds,opt)

    test(model, test_dl, user_embeds,ques_embeds)
