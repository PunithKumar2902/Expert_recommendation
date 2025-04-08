import os
import torch
import argparse
import Constants as C

from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence

from utils import *

from torch.utils.data import DataLoader
from Src.Models import Ranking_model

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

            # print("===================epoch=======================")
            # print("user ids ", users.device)
            # print("top user ids ", top_users.size())
            # print("ques tag ids ", questions.size())
            # print("==========================================")
            
            loss = model(users, top_users, questions,batch['UserId'])

            tot_loss+=loss

            loss.backward()

            #remove

            # # After loss.backward(), check the gradients of the layers
            # for name, param in model.named_parameters():
            #     if param.grad is not None:
            #         print(f'Gradient for {name}: {param.grad.norm()}')
            #     else:
            #         print(f'No gradient for {name}')

            #remove

            optimizer.step()
            # scheduler.step()

        print(f"Epoch {epoch+1} completed with total loss {tot_loss}")
        print(f"Epoch {epoch+1} completed with Avg loss {tot_loss/len(dl)}\n")
        with open(f"outputs/{C.DATASET}_output.txt", "a") as file:
            file.write(f"\nEpoch {epoch+1} completed with total loss {tot_loss}\n")
            file.write(f"Epoch {epoch+1} completed with Avg loss {tot_loss/len(dl)}\n")

        test(model, test_dl, user_embeds,ques_embeds)

def test(model, dl, user_embeds, ques_embeds):

    k=5
    #MRR, hit_K, prec_1 = 0, 0, 0

    print("===============================================================")
    print("Test ques: ",len(dl))
    print("===============================================================")
    MRR, hit_K, prec_1, hit_3, hit_4, hit_2, hit_1 = 0, 0, 0, 0, 0, 0, 0

    model.eval()
    with torch.no_grad():
        
        for batch in tqdm(dl, mininterval = 2, desc='  - (Testing) ', leave=False):
            
            users, top_users, questions = get_embeddings(batch, user_embeds, ques_embeds)

            if len(list(users.size()))>2:
                users = users.squeeze(0)

            #remove
            # print("1 : ",users.size())
            # print("2 : ",questions.size())
            #remove

            #uncomment
            # scores = model.test(users,questions)
            #uncomment

            #remove
            scores = torch.matmul(users,questions.T).squeeze()
            #remove

            #remove
            # random_scores = torch.randn(users.size(0), 1)
            #remove

            #remove
            # with open("outputs/test_random_scores.txt", "a") as file:
            #     file.write(str(users.size()) +"\t\t"+str(scores.size()) + "\n")
            # #remove
            
            user_batch_ids = batch['UserId'].clone().detach()
            top_user_ids = batch['Top_userId'].clone().detach()

            # R_R, hit, prec = performance_metrics(user_batch_ids,scores,top_user_ids,k) 

            #remove
            # punith = performance_metrics(user_batch_ids,random_scores,top_user_ids,k) 
            # print("=====================================================================")
            # print("hey : ",punith) 
            # print("=====================================================================") 
            #remove

            # R_R, hit, prec = performance_metrics(user_batch_ids,random_scores,top_user_ids,k) 
            R_R, hit, prec = performance_metrics(user_batch_ids,scores,top_user_ids,k) 

            MRR += R_R
            hit_K += hit
            prec_1 += prec

        # MRR, hit_K, prec_1 = MRR / batch_len, hit_K / test_batch_len, prec_1 / test_batch_len

        print("===============================================================")
        print("MRR : ",MRR)
        print("hit_K : ",hit_K)
        print("prec_1 : ",prec_1) 
        # print("hit_1 : ",hit_1)
        # print("hit_2 : ",hit_2)
        # print("hit_3 : ",hit_3)
        # print("hit_4 : ",hit_4)
        print("===============================================================")

        print("===============================================================")
        print("MRR : ",MRR/len(dl))
        print("hit_K : ",hit_K/len(dl))
        print("prec_1 : ",prec_1/len(dl)) 
        # print("hit_1 : ",hit_1/len(dl))
        # print("hit_2 : ",hit_2/len(dl))
        # print("hit_3 : ",hit_3/len(dl))
        # print("hit_4 : ",hit_4/len(dl))
        print("===============================================================")

        with open(f"outputs/{C.DATASET}_output.txt", "a") as file:
            file.write("===============================================================\n")
            file.write(f"MRR : {MRR}\n")
            file.write(f"hit_K : {hit_K}\n")
            file.write(f"prec_1 : {prec_1}\n")
            # file.write(f"hit_1 : {hit_1}\n")
            # file.write(f"hit_2 : {hit_2}\n")
            # file.write(f"hit_3 : {hit_3}\n")
            # file.write(f"hit_4 : {hit_4}\n")
            file.write("===============================================================\n")

            file.write("===============================================================\n")
            file.write(f"MRR : {MRR/len(dl)}\n")
            file.write(f"hit_K : {hit_K/len(dl)}\n")
            file.write(f"prec_1 : {prec_1/len(dl)}\n")
            # file.write(f"hit_1 : {hit_1/len(dl)}\n")
            # file.write(f"hit_2 : {hit_2/len(dl)}\n")
            # file.write(f"hit_3 : {hit_3/len(dl)}\n")
            # file.write(f"hit_4 : {hit_4/len(dl)}\n")
            file.write("===============================================================\n")


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    opt = parser.parse_args()
    opt.device = torch.device('cuda')

    opt.epoch = 5
    opt.n_layers = 1  

    #uncomment
    # opt.batch_size = 32 
    #uncomment

    #delete
    opt.batch_size = 8
    #delete

    opt.dropout = 0.6
    opt.smooth = 0.06
    opt.n_head = 1
    opt.d_model = C.EMB_size

    opt.lr = 0.0001

    embed_path = f"results/{C.DATASET}/{C.DATASET}"
    data_path=f'/u/student/2023/cs23mtech11032/Punith/Expert_recommendation/user_pre_training/data/{C.DATASET}'

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

    train_data, test_data = give_data(data_path)

    train_data = RankingDataset(train_data)
    test_data = RankingDataset(test_data)

    train_dl = DataLoader(train_data, batch_size=8, shuffle=True,collate_fn = collate_fn)
    test_dl = DataLoader(test_data, batch_size=1, shuffle=True)

    model = Ranking_model(opt.d_model,opt.d_model//512+1,1)

    model = model.to(opt.device)

    # train(model, train_dl, user_embeds, ques_embeds,opt)

    test(model, test_dl, user_embeds,ques_embeds)
