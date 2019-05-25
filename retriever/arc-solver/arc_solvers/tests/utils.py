import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import *

import numpy as np
import argparse

# Constant
MAX_HITS = 8
MAX_LEN = 120

### embedding

NO_EMBEDDING_ERR = "Embedding {} not in EMBEDDING_REGISTRY! Available embeddings are {}"

EMBEDDING_REGISTRY = {}


def RegisterEmbedding(name):
    """Registers a dataset."""

    def decorator(f):
        EMBEDDING_REGISTRY[name] = f
        return f
    return decorator


# Depending on arg, return embeddings
def get_embedding_tensor(args):
    if args.embedding not in EMBEDDING_REGISTRY:
        raise Exception(
            NO_EMBEDDING_ERR.format(args.embedding, EMBEDDING_REGISTRY.keys()))

    if args.embedding in EMBEDDING_REGISTRY:
        embeddings, word_to_indx = EMBEDDING_REGISTRY[args.embedding](args)

    args.embedding_dim = embeddings.shape[1]

    return embeddings, word_to_indx


@RegisterEmbedding('beer')
def getBeerEmbedding(args):
    embedding_path='raw_data/beer_review/review+wiki.filtered.200.txt.gz'
    lines = []
    with gzip.open(embedding_path) as file:
        lines = file.readlines()
        file.close()
    embedding_tensor = []
    word_to_indx = {}
    for indx, l in enumerate(lines):
        word, emb = l.split()[0], l.split()[1:]
        vector = [float(x) for x in emb ]
        if indx == 0:
            embedding_tensor.append( np.zeros( len(vector) ) )
        embedding_tensor.append(vector)
        word_to_indx[word] = indx+1
    embedding_tensor = np.array(embedding_tensor, dtype=np.float32)
    return embedding_tensor, word_to_indx

@RegisterEmbedding('glove')
def getGloveEmbedding(args):
    embedding_path='/data2/jianmo/data/glove/glove.6B.300d.txt'
    lines = []
    with open(embedding_path) as file:
        lines = file.readlines()
        file.close()
    embedding_tensor = []
    word_to_indx = {}
    for indx, l in enumerate(lines):
        word, emb = l.split()[0], l.split()[1:]
        if not len(emb) == 300:
            continue
        vector = [float(x) for x in emb ]
        if indx == 0:
            embedding_tensor.append( np.zeros( len(vector) ) )
        embedding_tensor.append(vector)
        word_to_indx[word] = indx+1
    embedding_tensor = np.array(embedding_tensor, dtype=np.float32)
    return embedding_tensor, word_to_indx

@RegisterEmbedding('pathology')
def getPathologyEmbedding(args):
    embedding_path = 'pickle_files/embeddings.p'
    word_to_indx_path = 'pickle_files/vocabIndxDict.p'
    embedding_tensor = pickle.load(open(embedding_path,'rb'))
    word_to_indx = pickle.load(open(word_to_indx_path,'rb'))
    ## Add 0 embed at indx 0
    embedding_tensor = np.vstack( [np.zeros( (1, embedding_tensor.shape[1])), embedding_tensor ]).astype( np.float32)
    for word in word_to_indx:
        word_to_indx[word] += 1
    return embedding_tensor, word_to_indx


def get_indices_tensor_rl(text_arrs, word_to_indx, max_length):
    '''
    -text_arrs: array of (array of word tokens)
    -word_to_indx: mapping of word -> index
    -max length of return tokens

    returns tensor of same size as text with each words corresponding
    index
    '''
    
    text_indices = []
    x_oovs = []
    for text_arr in text_arrs:
        nil_indx = 0
        text_indx = [ word_to_indx[x] if x in word_to_indx else nil_indx for x in text_arr][:max_length]
        if len(text_indx) < max_length:
            text_indx.extend( [nil_indx for _ in range(max_length - len(text_indx))])
        text_indices.append(text_indx)

        x_oov = [ 1 if x not in word_to_indx else 0 for x in text_arr][:max_length]
        if len(x_oov) < max_length:
            x_oov.extend( [0 for _ in range(max_length - len(x_oov))])   
        x_oovs.append(x_oov)      
                
    x =  torch.LongTensor([text_indices])
    x_oovs =  torch.LongTensor([x_oovs])
    
    return x, x_oovs

### unknown word and padding word all set to nil_indx?!
def get_indices_tensor(text_arr, word_to_indx, max_length):
    '''
    -text_arr: array of word tokens
    -word_to_indx: mapping of word -> index
    -max length of return tokens

    returns tensor of same size as text with each words corresponding
    index
    '''
    nil_indx = 0
    text_indx = [ word_to_indx[x] if x in word_to_indx else nil_indx for x in text_arr][:max_length]
    if len(text_indx) < max_length:
        text_indx.extend( [nil_indx for _ in range(max_length - len(text_indx))])

    x =  torch.LongTensor([text_indx])

    return x



### training utils

def get_x_indx(batch, args, eval_model):
    if eval_model:
        with torch.no_grad():
            x_indx = Variable(batch['x_indx'])
    else:
        x_indx = Variable(batch['x_indx'])
    return x_indx

def get_optimizer(models, args):
    '''
        -models: List of models (such as Generator, classif, memory, etc)
        -args: experiment level config

        returns: torch optimizer over models
    '''
    params = []
    for model in models:
        params.extend([param for param in model.parameters() if param.requires_grad])
    return torch.optim.Adam(params, lr=args.lr,  weight_decay=args.weight_decay)

def init_metrics_dictionary(modes):
    '''
    Create dictionary with empty array for each metric in each mode
    '''
    epoch_stats = {}
    metrics = [
        'loss', 'score' ]
        # 'obj_loss', 'k_selection_loss',
        # 'k_continuity_loss', 'metric', 'confusion_matrix']
    for metric in metrics:
        for mode in modes:
            key = "{}_{}".format(mode, metric)
            epoch_stats[key] = []

    return epoch_stats


def get_gen_path(model_path):
    '''
        -model_path: path of encoder model

        returns: path of generator
    '''
    return '{}.gen'.format(model_path)


def collate_epoch_stat(stat_dict, epoch_details, mode, args):
    '''
        Update stat_dict with details from epoch_details and create
        log statement

        - stat_dict: a dictionary of statistics lists to update
        - epoch_details: list of statistics for a given epoch
        - mode: train, dev or test
        - args: model run configuration

        returns:
        -stat_dict: updated stat_dict with epoch details
        -log_statement: log statement sumarizing new epoch

    '''
    log_statement_details = ''
    for metric in epoch_details:
        loss = epoch_details[metric]
        stat_dict['{}_{}'.format(mode, metric)].append(loss)

        log_statement_details += ' -{}: {}'.format(metric, loss)

    log_statement = '{}\n--'.format(log_statement_details )

    return stat_dict, log_statement


### parse args

def parse_args():
    parser = argparse.ArgumentParser(description='Rationale-Net Classifier')

    #setup
    parser.add_argument('--train', action='store_true', default=False, help='Whether or not to train model')
    parser.add_argument('--test', action='store_true', default=False, help='Whether or not to run model on test set')
    # device
    parser.add_argument('--cuda', action='store_true', default=False, help='enable the gpu' )
    parser.add_argument('--num_gpus', type=int, default=1, help='Num GPUs to use.')
    parser.add_argument('--debug_mode', action='store_true', default=False, help='debug mode' )
    parser.add_argument('--query_mode', type=str, default='hypothesis', help='query mode' )
    
    # learning
    parser.add_argument('--init_lr', type=float, default=0.001, help='initial learning rate [default: 0.001]')
    parser.add_argument('--epochs', type=int, default=256, help='number of epochs for train [default: 256]')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size for training [default: 64]')
    parser.add_argument('--max_batch_size', type=int, default=2000, help='Max batch size to fit through gpu')
    parser.add_argument('--patience', type=int, default=10, help='Num epochs of no dev progress before half learning rate [default: 10]')
    parser.add_argument('--erate', type=float, default=0.01, help='Weight of cross-entropy loss')

    #paths
    parser.add_argument('--save_dir', type=str, default='snapshot', help='where to save the snapshot')
    parser.add_argument('--results_path', type=str, default='/data2/jianmo/reading_comprehension/output/rl', help='where to dump model config and epoch stats')
    parser.add_argument('--summary_path', type=str, default='results/summary.csv', help='where to dump model config and epoch stats')
    parser.add_argument('--snapshot', type=str, default=None, help='filename of model snapshot to load[default: None]')
    # data loading
    parser.add_argument('--num_workers' , type=int, default=1, help='num workers for data loader')
    # model
    parser.add_argument('--model_form', type=str, default='cnn', help="Form of model, i.e cnn, rnn, etc.")
    parser.add_argument('--hidden_dim', type=int, default=100, help="Dim of hidden layer")
    parser.add_argument('--num_layers', type=int, default=1, help="Num layers of model_form to use")
    parser.add_argument('--dropout', type=float, default=0.1, help='the probability for dropout [default: 0.5]')
    parser.add_argument('--weight_decay', type=float, default=1e-3, help='L2 norm penalty [default: 1e-3]')
    parser.add_argument('--filter_num', type=int, default=100, help='number of each kind of kernel')
    parser.add_argument('--filters', type=str, default=[3,4,5], help='comma-separated kernel size to use for convolution')

    parser.add_argument('--dataset', default='arc', help='choose which dataset to run on')
    parser.add_argument('--embedding', default='glove', help='choose what embeddings to use')
    parser.add_argument('--embedding_dim' , type=int, default=300, help='dimension of ')

    args = parser.parse_args()

    # update args and print

    print("\nParameters:")
    for attr, value in sorted(args.__dict__.items()):
        print("\t{}={}".format(attr.upper(), value))

    return args
