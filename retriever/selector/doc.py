import torch
import numpy as np
import time

from utils import vocab, pos_vocab, ner_vocab, rel_vocab
from allennlp.modules.elmo import Elmo, batch_to_ids

class Example:

    def __init__(self, input_dict, elmo=None, use_science_term=False):
        self.id = input_dict['id']
        self.question = input_dict['q_words']
        self.choice = input_dict['c_words']
        self.q_pos = input_dict['q_pos']
        self.q_ner = input_dict['q_ner']
        assert len(self.q_pos) == len(self.question.split()), (self.q_pos, self.question)
   
        if use_science_term:
            self.features = np.stack([input_dict['in_c'], input_dict['lemma_in_c'], input_dict['tf'], input_dict['q_is_science_term']], 1)
        else:        
            self.features = np.stack([input_dict['in_c'], input_dict['lemma_in_c'], input_dict['tf']], 1)

        assert len(self.features) == len(self.question.split())
        self.label = input_dict['label']
        self.label_tensor = torch.ByteTensor(self.label)
  
        # self.answer = input_dict['answer'] # not used... 
     
        # convert to idx
        # if use elmo, passage/question/choice text are no longer need to convert to idx
        if elmo == None:
            self.q_tensor = torch.LongTensor([vocab[w] for w in self.question.split()])
            self.c_tensor = torch.LongTensor([vocab[w] for w in self.choice.split()])
            
        self.q_pos_tensor = torch.LongTensor([pos_vocab[w] for w in self.q_pos])
        self.q_ner_tensor = torch.LongTensor([ner_vocab[w] for w in self.q_ner])
        self.features = torch.from_numpy(self.features).type(torch.FloatTensor)
        self.q_c_relation = torch.LongTensor([rel_vocab[r] for r in input_dict['q_c_relation']])
        self.q_is_cand = torch.LongTensor(input_dict['q_is_cand'])

    def __str__(self):
        return 'Question: {}\n Choices: {}, Label: {}'.format(self.question, self.choice, self.label)

    
def _to_indices_and_mask_elmo(batch_text, elmo):
    activations, mask = elmo.batch_to_embeddings(batch_text)
    activations = torch.sum(activations, dim=1) 
    mask = mask != 1
    return activations, mask


def _to_indices_and_mask(batch_tensor, need_mask=True):
    mx_len = max([t.size(0) for t in batch_tensor])
    batch_size = len(batch_tensor)
    indices = torch.LongTensor(batch_size, mx_len).fill_(0)
    if need_mask: # initialize as 1
        mask = torch.ByteTensor(batch_size, mx_len).fill_(1)
    for i, t in enumerate(batch_tensor):
        indices[i, :len(t)].copy_(t)
        if need_mask:
            mask[i, :len(t)].fill_(0) # fill non-padding position as 0...
    if need_mask:
        return indices, mask
    else:
        return indices

def _to_feature_tensor(features):
    mx_len = max([f.size(0) for f in features])
    batch_size = len(features)
    f_dim = features[0].size(1)
    f_tensor = torch.FloatTensor(batch_size, mx_len, f_dim).fill_(0)
    for i, f in enumerate(features):
        f_tensor[i, :len(f), :].copy_(f)
    return f_tensor


def batchify(batch_data, elmo=None):
    q_pos = _to_indices_and_mask([ex.q_pos_tensor for ex in batch_data], need_mask=False)
    q_ner = _to_indices_and_mask([ex.q_ner_tensor for ex in batch_data], need_mask=False)
    q_c_relation = _to_indices_and_mask([ex.q_c_relation for ex in batch_data], need_mask=False)
    f_tensor = _to_feature_tensor([ex.features for ex in batch_data])
    q_is_cand = _to_indices_and_mask([ex.q_is_cand for ex in batch_data], need_mask=False)
    
    if elmo==None:
        q, q_mask = _to_indices_and_mask([ex.q_tensor for ex in batch_data])
        c, c_mask = _to_indices_and_mask([ex.c_tensor for ex in batch_data])
    else:
        # do batch_to_ids later
        q, q_mask = _to_indices_and_mask_elmo([ex.question.split() for ex in batch_data], elmo)
        c, c_mask = _to_indices_and_mask_elmo([ex.choice.split() for ex in batch_data], elmo)
    
    y, y_mask = _to_indices_and_mask([ex.label_tensor for ex in batch_data])    
        
    return q, q_pos, q_ner, q_mask, c, c_mask, f_tensor, q_c_relation, q_is_cand, y, y_mask
