import os
import sys
import time
# import optuna
import argparse
import numpy as np
import Constants as C

# from optuna.trial import TrialState

from src.Models import Model
from preprocess.Dataset import Dataset as dataset

import torch
import torch.optim as optim

from src import metric, Utils

from tqdm import tqdm

import random

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU

    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_epoch(model, user_dl, optimizer, opt):
    """ Epoch operation in training phase. """

    model.train()
    [pre, rec, map_, ndcg] = [[[] for i in range(4)] for j in range(4)]
    for batch in tqdm(user_dl, mininterval=2, desc='  - (Training)   ', leave=False):
        optimizer.zero_grad()

        """ prepare data """
        event_type, event_time, test_label = map(lambda x: x.to(opt.device), batch)

        """ forward """
        prediction, users_embeddings = model(event_type)

        """ compute metric """
        metric.pre_rec_top(pre, rec, map_, ndcg, prediction, test_label, event_type)

        """ backward """
        loss = Utils.type_loss(prediction, event_type, event_time, test_label, opt)

        loss.backward(retain_graph=True)
        """ update parameters """
        optimizer.step()

    results_np = map(lambda x: [np.around(np.mean(i), 5) for i in x], [pre, rec, map_, ndcg])
    return results_np


def eval_epoch(model, user_valid_dl, opt):
    """ Epoch operation in evaluation phase. """

    model.eval()
    [pre, rec, map_, ndcg] = [[[] for i in range(4)] for j in range(4)]
    with torch.no_grad():
        for batch in tqdm(user_valid_dl, mininterval=2,
                          desc='  - (Validation) ', leave=False):
            """ prepare test data """
            event_type, event_time, test_label = map(lambda x: x.to(opt.device), batch)

            """ forward """
            prediction, users_embeddings = model(event_type)  # X = (UY+Z) ^ T

            """ compute metric """
            metric.pre_rec_top(pre, rec, map_, ndcg, prediction, test_label, event_type)

    results_np = map(lambda x: [np.around(np.mean(i), 5) for i in x], [pre, rec, map_, ndcg])
    return results_np


# def train(trial, model, data, optimizer, scheduler, opt):
def train(model, data, optimizer, scheduler, opt):
    """ Start training. """
    (user_valid_dl, user_dl) = data

    best_ = [np.zeros(4) for i in range(4)]
    for epoch_i in range(opt.epoch):
        print('[ Epoch', epoch_i + 1, ']')

        with open(f"outputs/{C.DATASET}_output.txt", "a") as file:
            file.write(f'\n[ Epoch{epoch_i + 1}]')

        np.set_printoptions(formatter={'float': '{: 0.5f}'.format})
        start = time.time()
        [pre, rec, map_, ndcg] = train_epoch(model, user_dl, optimizer, opt)
        print('\r(Training)  P@k:{pre},    R@k:{rec}, \n'
              '(Training)map@k:{map_}, ndcg@k:{ndcg}, '
              'elapse:{elapse:3.3f} min'
              .format(elapse=(time.time() - start) / 60, pre=pre, rec=rec, map_=map_, ndcg=ndcg))

        with open(f"outputs/{C.DATASET}_output.txt", "a") as file:
            file.write('\r(Training)  P@k:{pre},    R@k:{rec}, \n'
              '(Training)map@k:{map_}, ndcg@k:{ndcg}, '
              'elapse:{elapse:3.3f} min'
              .format(elapse=(time.time() - start) / 60, pre=pre, rec=rec, map_=map_, ndcg=ndcg))

        start = time.time()
        [pre, rec, map_, ndcg] = eval_epoch(model, user_valid_dl, opt)
        print('\n\r(Test)  P@k:{pre},    R@k:{rec}, \n'
              '(Test)map@k:{map_}, ndcg@k:{ndcg}, '
              'elapse:{elapse:3.3f} min'
              .format(elapse=(time.time() - start) / 60, pre=pre, rec=rec, map_=map_, ndcg=ndcg))
        with open(f"outputs/{C.DATASET}_output.txt", "a") as file:
            file.write('\r(Test)  P@k:{pre},    R@k:{rec}, \n'
              '(Test)map@k:{map_}, ndcg@k:{ndcg}, '
              'elapse:{elapse:3.3f} min'
              .format(elapse=(time.time() - start) / 60, pre=pre, rec=rec, map_=map_, ndcg=ndcg))

        scheduler.step()

        if best_[-1][1] < ndcg[1]: best_ = [pre, rec, map_, ndcg]
        
        # trial.report(ndcg[1], epoch_i)

        # # Handle pruning based on the intermediate value.
        # if trial.should_prune():
        #     raise optuna.exceptions.TrialPruned()

    os.makedirs(f'results/{C.DATASET}', exist_ok=True)

    torch.save(model.state_dict(),f'results/{C.DATASET}/{C.DATASET}.pth')

    with open('results/beta_lambda_{}.log'.format(C.DATASET), 'a') as f:
        f.write("%s Beta:%.4f Lambda:%.4f " % (C.DATASET, opt.lambda_, opt.delta))
        f.write("P@k:{pre}, R@k:{rec}, {map_}, ndcg@k:{ndcg}\n"
                .format(pre=best_[0], rec=best_[1], map_=best_[2], ndcg=best_[3]))
        f.close()
    
    return best_[-1][1]

# def main(trial):
def main():
    """ Main function. """
    parser = argparse.ArgumentParser()
    opt = parser.parse_args()
    opt.device = torch.device('cuda')
    set_seed(42)
    
    # >> optuna setting for tuning hyperparameters
    # opt.n_layers = trial.suggest_int('n_layers', 1,5, )
    # opt.n_head = trial.suggest_int('n_head', 1, 5, step=1)
    # opt.d_model = trial.suggest_int('d_model', 128, 512, step=64)
    # opt.epoch = trial.suggest_int('epoch', 10, 30, step=2)
    # opt.batch_size = trial.suggest_int('batch_size', 8, 16, step=2)

    # opt.dropout = trial.suggest_float('dropout_rate', 0.5, 0.7)
    # opt.smooth = trial.suggest_float('smooth', 1e-2, 1e-1)
    # opt.lr = trial.suggest_float('learning_rate', 0.00008, 0.0002)


    # opt.lambda_, opt.delta = trial.suggest_float('lambda', 0.1, 4), trial.suggest_float('delta', 0.1, 4)

    opt.n_layers = 1
    opt.n_head = 1
    opt.d_model = C.EMB_size #128 #1024 #448
    opt.epoch = 6
    opt.batch_size = 8

    opt.dropout = 0.62
    opt.smooth = 0.06
    opt.lr = 0.00001 #0.00016

    opt.lambda_, opt.delta = 1,4  #1.7,1.7

    print('[Info] parameters: {}'.format(opt))
    """ prepare model """
    model = Model(
        num_types=C.TAG_NUMBER,
        d_model=opt.d_model,
        n_layers=opt.n_layers,
        n_head=opt.n_head,
        dropout=opt.dropout,
        device=opt.device
    )
    model = model.cuda()

    """ loading data"""
    print('[Info] Loading data...')
    ds = dataset()
    user_dl = ds.get_user_dl(opt.batch_size)
    user_valid_dl = ds.get_user_valid_dl(opt.batch_size)
    data = (user_valid_dl, user_dl)

    """ optimizer and scheduler """
    parameters = [{'params': model.parameters(), 'lr': opt.lr},]
    optimizer = torch.optim.Adam(parameters)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 10, gamma=0.5)

    """ train the model """
    return train(model, data, optimizer, scheduler, opt)
    # return train(trial, model, data, optimizer, scheduler, opt)

if __name__ == '__main__':
    
    main()
    # study = optuna.create_study(direction="maximize")
    # study.optimize(main, n_trials=100)

    # pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    # complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    # print("Study statistics: ")
    # print("  Number of finished trials: ", len(study.trials))
    # print("  Number of pruned trials: ", len(pruned_trials))
    # print("  Number of complete trials: ", len(complete_trials))

    # print("Best trial:")
    # trial = study.best_trial

    # print("  Value: ", trial.value)

    # print("  Params: ")
    # for key, value in trial.params.items():
    #     print("    {}: {}".format(key, value))