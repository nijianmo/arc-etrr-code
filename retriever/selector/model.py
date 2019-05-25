import logging
import copy
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np
import time

from utils import vocab, flatten
from doc import batchify
from trian import simpleModel
from allennlp.commands.elmo import ElmoEmbedder
from sklearn.metrics import f1_score, precision_recall_fscore_support

logger = logging.getLogger()

class Model:

    def __init__(self, args):
        self.args = args
        self.batch_size = args.batch_size
        self.finetune_topk = args.finetune_topk
        self.lr = args.lr
        self.use_cuda = (args.use_cuda == True) and torch.cuda.is_available()
        print('Use cuda:', self.use_cuda)
        if self.use_cuda:
            torch.cuda.set_device(int(args.gpu))
        # self.network = TriAN(args)
        self.network = simpleModel(args)
        self.init_optimizer()
        # load pretrained model
        if args.pretrained:
            print('Load pretrained model from %s...' % args.pretrained)
            self.load(args.pretrained)
        else:
            if args.use_elmo==False:
                self.load_embeddings(vocab.tokens(), args.embedding_file)
        if args.use_elmo==False:
            self.network.register_buffer('fixed_embedding', self.network.embedding.weight.data[self.finetune_topk:].clone())
            self.elmo = None
        else:
            options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
            weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
            self.elmo = ElmoEmbedder(options_file, weight_file, cuda_device=0)          
        if self.use_cuda:
            self.network.cuda()
        print(self.network)
        self._report_num_trainable_parameters()
        
        self.crit = torch.nn.BCEWithLogitsLoss(reduce=False, pos_weight=torch.FloatTensor([args.pos_weight]).cuda())
 
    def _report_num_trainable_parameters(self):
        num_parameters = 0
        for p in self.network.parameters():
            if p.requires_grad:
                sz = list(p.size())
                if sz[0] == len(vocab):
                    sz[0] = self.finetune_topk
                num_parameters += np.prod(sz)
        print('Number of parameters: ', num_parameters)

    def train(self, train_data):
        self.network.train()
        self.updates = 0
        iter_cnt, num_iter = 0, (len(train_data) + self.batch_size - 1) // self.batch_size
        for batch_input in self._iter_data(train_data):
            
            feed_input = [x for x in batch_input[:-3]]  
            q_mask = feed_input[3]
            q_is_cand, y, y_mask = batch_input[-3], batch_input[-2], batch_input[-1] # now y has mask 

            pred_proba = self.network(*feed_input)
            # pred_proba = pred_proba.view(-1, 2)
            pred_proba = pred_proba.view(-1, 1)
            y = y.view(-1, 1)
            q_mask = q_mask.view(-1, 1)
            
            # loss = F.cross_entropy(pred_proba, y, reduce=False)
            loss = self.crit(pred_proba, y.float())
            
            loss.data.masked_fill_(q_mask.data, 0)
            loss = torch.sum(loss) / torch.sum(q_mask.eq(0)).float() # take average
            
            self.optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm(self.network.parameters(), self.args.grad_clipping)

            # Update parameters
            self.optimizer.step()
            if self.args.use_elmo==False:
                self.network.embedding.weight.data[self.finetune_topk:] = self.network.fixed_embedding
            self.updates += 1
            iter_cnt += 1

            if self.updates % 20 == 0:
            #if self.updates % 10 == 0:
                print('Iter: %d/%d, Loss: %f' % (iter_cnt, num_iter, loss.item()))
        self.scheduler.step()
        print('LR:', self.scheduler.get_lr()[0])

    def evaluate(self, dev_data, debug=False, eval_train=False):
        if len(dev_data) == 0:
            return -1.0, -1.0, -1.0, -1.0
        self.network.eval()
        prediction, gold = [], []
        dev_data = sorted(dev_data, key=lambda ex: ex.id)
        for batch_input in self._iter_data(dev_data):
            
            feed_input = [x for x in batch_input[:-3]]  
            q_mask = feed_input[3]
            q_is_cand, y, y_mask = batch_input[-3], batch_input[-2], batch_input[-1] # now y has mask 

            pred_proba = self.network(*feed_input)
            length = torch.sum(q_mask.eq(0), dim=1)
            
            #pred = pred_proba.argmax(dim=-1) # predict
            pred = pred_proba.squeeze(2) > 0.5 
            
            y = list(y.data.cpu().numpy())
            pred = list(pred.data.cpu().numpy())
            length = list(length.data.cpu())
            q_is_cand = list(q_is_cand.data.cpu().numpy())
            
            assert(len(pred) == len(length))
            
            # when evaluate, only consider non-stop and non-punc words
            pred_list, gold_list = [], []
            for yl,pl,l,cl in zip(y,pred,length,q_is_cand):
                yl_essential, pl_essential= [], []
                for yw, pw, c in zip(yl[:l], pl[:l], cl[:l]): 
                    if c == 1:
                        yl_essential.append(yw)
                        pl_essential.append(pw)
                    
                gold_list.append(yl_essential)
                pred_list.append(pl_essential)
                
            prediction.extend(pred_list) # extend samples from different batch
            gold.extend(gold_list)
           
        prediction_flat = flatten(prediction)
        gold_flat = flatten(gold)
        
        correct = sum(np.array(prediction_flat) == np.array(gold_flat))
        total = len(prediction_flat)
        acc = 100 * correct / total
        precision, recall, f1, _ = precision_recall_fscore_support(gold_flat, prediction_flat)
        
        return acc, precision[1], recall[1], f1[1]

    def predict(self, test_data):
        # DO NOT SHUFFLE test_data
        self.network.eval()
        prediction = []
        for batch_input in self._iter_data(test_data):
            
            feed_input = [x for x in batch_input[:-3]]  
            q_mask = feed_input[3]
            q_is_cand, y, y_mask = batch_input[-3], batch_input[-2], batch_input[-1] # now y has mask 

            pred_proba = self.network(*feed_input)
            length = torch.sum(q_mask.eq(0), dim=1)
            q_is_cand = list(q_is_cand.data.cpu().numpy())

            #pred = pred_proba.argmax(dim=-1) # predict
            pred = pred_proba.squeeze(2) > 0.5 
            
            pred = list(pred.data.cpu().numpy())
            length = list(length.data.cpu())
            
            assert(len(pred) == len(length))
            
            # when evaluate, only consider non-stop and non-punc words
            pred_list = []
            for pl,l,cl in zip(pred,length,q_is_cand):
                pl_essential= []
                for pw, c in zip(pl[:l], cl[:l]): 
                    if c == 1:
                        pl_essential.append(pw)
                
                pred_list.append(pl_essential)
            
            pred_list = []
            for pl,l in zip(pred,length):
                pred_list.append(pl[:l])
                
            prediction.extend(pred_list) # extend samples from different batch
            
        return prediction

    def _iter_data(self, data):
        num_iter = (len(data) + self.batch_size - 1) // self.batch_size
        for i in range(num_iter):
            start_idx = i * self.batch_size
            batch_data = data[start_idx:(start_idx + self.batch_size)]
            
            # convert a batch into tensors
            # batch_input = batchify(batch_data)
            batch_input = batchify(batch_data, self.elmo)
    
            # Transfer to GPU
            if self.use_cuda:
                batch_input = [Variable(x.cuda(async=True)) for x in batch_input]
            else:
                batch_input = [Variable(x) for x in batch_input]

            yield batch_input

    def load_embeddings(self, words, embedding_file):
        """Load pretrained embeddings for a given list of words, if they exist.

        Args:
            words: iterable of tokens. Only those that are indexed in the
              dictionary are kept.
            embedding_file: path to text file of embeddings, space separated.
        """
        words = {w for w in words if w in vocab} 
        logger.info('Loading pre-trained embeddings for %d words from %s' %
                    (len(words), embedding_file))
        embedding = self.network.embedding.weight.data

        # When normalized, some words are duplicated. (Average the embeddings).
        vec_counts = {}
        with open(embedding_file) as f:
            for line in f:
                parsed = line.rstrip().split(' ')
                assert(len(parsed) == embedding.size(1) + 1)
                w = vocab.normalize(parsed[0])
                if w in words:
                    vec = torch.Tensor([float(i) for i in parsed[1:]])
                    if w not in vec_counts:
                        vec_counts[w] = 1
                        embedding[vocab[w]].copy_(vec)
                    else:
                        logging.warning('WARN: Duplicate embedding found for %s' % w)
                        vec_counts[w] = vec_counts[w] + 1
                        embedding[vocab[w]].add_(vec)

        for w, c in vec_counts.items():
            embedding[vocab[w]].div_(c)

        logger.info('Loaded %d embeddings (%.2f%%)' %
                    (len(vec_counts), 100 * len(vec_counts) / len(words)))

    def init_optimizer(self):
        parameters = [p for p in self.network.parameters() if p.requires_grad]
        if self.args.optimizer == 'sgd':
            self.optimizer = optim.SGD(parameters, self.lr,
                                       momentum=0.4,
                                       weight_decay=0)
        elif self.args.optimizer == 'adamax':
            self.optimizer = optim.Adamax(parameters,
                                        lr=self.lr, 
                                        weight_decay=self.args.weight_decay)
        else:
            raise RuntimeError('Unsupported optimizer: %s' %
                               self.args.optimizer)
        self.scheduler = lr_scheduler.MultiStepLR(self.optimizer, milestones=[10, 15], gamma=0.5)

    def save(self, ckt_path):
        state_dict = copy.copy(self.network.state_dict())
        if 'fixed_embedding' in state_dict:
            state_dict.pop('fixed_embedding')
        params = {'state_dict': state_dict}
        torch.save(params, ckt_path)

    def load(self, ckt_path):
        logger.info('Loading model %s' % ckt_path)
        saved_params = torch.load(ckt_path, map_location=lambda storage, loc: storage)
        state_dict = saved_params['state_dict']
        return self.network.load_state_dict(state_dict)

    def cuda(self):
        self.use_cuda = True
        self.network.cuda()
