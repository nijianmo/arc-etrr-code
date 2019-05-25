import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import *
import arc_solvers.tests.cnn as cnn
import arc_solvers.tests.utils as utils
import tqdm
import os

from arc_solvers.tests.es_search import EsSearch, EsHit
from arc_solvers.tests.rewards import *


def train_model(train_data, dev_data, gen, args, es_search):
    '''
    Train model and tune on dev set. If model doesn't improve dev performance within args.patience
    epochs, then halve the learning rate, restore the model to best and continue training.

    At the end of training, the function will restore the model to best dev version.

    returns epoch_stats: a dictionary of epoch level metrics for train and test
    returns model : best model from this call to train
    '''

    if args.cuda:
        gen = gen.cuda()

    args.lr = args.init_lr
    optimizer = utils.get_optimizer([gen], args)

    num_epoch_sans_improvement = 0
    epoch_stats = utils.init_metrics_dictionary(modes=['train', 'dev'])
    step = 0

    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=False)

    dev_loader = torch.utils.data.DataLoader(
        dev_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False)


    for epoch in range(1, args.epochs + 1):

        print("-------------\nEpoch {}:\n".format(epoch))
        for mode, dataset, loader in [('Train', train_data, train_loader), ('Dev', dev_data, dev_loader)]:
            train_model = mode == 'Train'
            print('{}'.format(mode))
            key_prefix = mode.lower()
            epoch_details, step, _ = run_epoch(
                dataset=dataset,
                data_loader=loader,
                train_model=train_model,
                model=gen,
                optimizer=optimizer,
                step=step,
                args=args,
                es_search=es_search)

            epoch_stats, log_statement = utils.collate_epoch_stat(epoch_stats, epoch_details, key_prefix, args)

            # Log  performance
            print(log_statement)

            if not train_model:
                print('---- Best Dev Loss is {:.4f}'.format(
                    min(epoch_stats['dev_loss'])))

        # Save model if beats best dev
        if min(epoch_stats['dev_loss']) == epoch_stats['dev_loss'][-1]:
            num_epoch_sans_improvement = 0
            if not os.path.isdir(args.save_dir):
                os.makedirs(args.save_dir)
            # Subtract one because epoch is 1-indexed and arr is 0-indexed
            epoch_stats['best_epoch'] = epoch - 1
            torch.save(gen, utils.get_gen_path(args.model_path))
        else:
            num_epoch_sans_improvement += 1

        if num_epoch_sans_improvement >= args.patience:
            print("Reducing learning rate")
            num_epoch_sans_improvement = 0
            gen = torch.load(utils.get_gen_path(args.model_path))

            if args.cuda:
                gen   = gen.cuda()
            args.lr *= .5
            optimizer = utils.get_optimizer([gen], args)

    # Restore model to best dev performance
    if os.path.exists(args.model_path):
        gen.cpu()
        gen = torch.load(utils.get_gen_path(args.model_path))

    return epoch_stats, gen



def run_epoch(dataset, data_loader, train_model, model, optimizer, step, args, es_search):
    '''
    Train model for one pass of train data, and return loss
    '''
    off = 1e-8
    
    eval_model = not train_model
    data_iter = data_loader.__iter__()

    losses = []
    total_scores = []

    if train_model:
        model.train()
    else:
        model.eval()

    num_batches_per_epoch = len(data_iter)
    if train_model:
        num_batches_per_epoch = min(len(data_iter), 10000)

    for _ in tqdm.tqdm(range(num_batches_per_epoch)):
        batch = data_iter.next()
        step += 1

        # text to index
        x_indx = utils.get_x_indx(batch, args, eval_model) # (Batch x N x T)
        x_oov = batch['x_oov'].squeeze(1) 
        y = batch['y']
        indices = batch['i'].data.cpu().numpy() # convert tensor to numpy array

        # x_indx: (Batch x 1 x N x T), N: number of choices, T: max length of each query
        if args.cuda:
            x_indx = x_indx.cuda()

        if train_model:
            optimizer.zero_grad()

        # feed into model, batch x 1 x N x T
        selections, probs = model(x_indx)  
        print(probs)

        cost = 0.
        #reward_last = 0

        # iterate each question
        preds = [] 
        scores = []

        batch_size = selections.size(0)
        for i in range(batch_size): 
            # N x T
            selection = selections[i]
            prob = probs[i]
            xi = dataset.text[indices[i]] # dict
            xi_oov = x_oov[i]
            yi = y[i]

            # convert binary selection into queries, search via ES, get answer, and compare with truth
            reward, mask, pred = get_search_metric(selection, xi, xi_oov, yi, es_search, dataset.query_mode)

            preds.append(pred)
            scores.append(reward)

            # r = reward - reward_last - bl # bl is the reward brought by queries that select all words?
            r = reward
                       
            # cost for the baseline
            # cost_bl = (r ** 2).sum()

            with torch.no_grad():
                r_ = reward
               
            logp = torch.log(prob * mask.float() + off)
            cost_i = r_ * (-logp) * selection
            curr_cost = cost_i.sum()

            # entropy regularization
            if args.erate > 0.:
                cost_ent = args.erate * (prob * logp).sum()
                curr_cost += cost_ent
            else:
                cost_ent = 0. * cost    

            cost += curr_cost
            # cost += cost_bl

            print("curr_cost={}, cost_ent={}".format(curr_cost, cost_ent))

        #reward_last = reward
        losses.append(cost.data.cpu().numpy() / batch_size)
        total_scores.append(np.mean(scores))

        if step % 10 == 0:
            max_display = 5
            for pred, score in zip(preds[:max_display], scores[:max_display]):
                print("query: {}, score: {}".format(pred, score))
            print("avg score in this batch: {}".format(np.mean(scores)))

        # update  
        if train_model:
            cost.backward()
            optimizer.step()

    # logging            
    epoch_stat = {
        'loss' : np.mean(losses),
        'score': np.mean(total_scores) 
    }     

    return epoch_stat, step, losses

