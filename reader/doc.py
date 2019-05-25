import torch
import numpy as np
import time

from utils import vocab, pos_vocab, ner_vocab, rel_vocab
from allennlp.modules.elmo import Elmo, batch_to_ids
from allennlp.commands.elmo import ElmoEmbedder

class Example:

    def __init__(self, input_dict, elmo=None):
        self.id = input_dict['id']
        self.passages = input_dict['d_words'] # passages: List
        self.question = input_dict['q_words']
        self.choices = input_dict['c_words'] # choices: List
        self.d_pos = input_dict['d_pos'] # List
        self.d_ner = input_dict['d_ner'] # List
        self.q_pos = input_dict['q_pos'] # List
        
        self.features = []
        for i in range(4):
            features = np.stack([input_dict['in_qs'][i], \
                                 input_dict['lemma_in_qs'][i], \
                                 input_dict['tfs'][i], \
                                 input_dict['in_cs'][i], \
                                 input_dict['lemma_in_cs'][i]], 1)                     
            self.features.append(features)

        self.q_features = np.stack([[1 if e else 0 for e in input_dict['q_es']]], 1)

        self.label = input_dict['label']

        # convert to idx
        # if use elmo, passage/question/choice text are no longer need to convert to idx
        if elmo == None:
            self.d_tensor = []
            for passage in self.passages:
                self.d_tensor.append(torch.LongTensor([vocab[w] for w in passage.split()])) # K x T: List
            self.q_tensor = torch.LongTensor([vocab[w] for w in self.question.split()])
            self.c_tensor = []
            for choice in self.choices:
                self.c_tensor.append(torch.LongTensor([vocab[w] for w in choice.split()])) # K x T: List of tensors
            
        self.d_pos_tensor = []
        self.d_ner_tensor = [] 
        for d_pos in self.d_pos:
            self.d_pos_tensor.append(torch.LongTensor([pos_vocab[w] for w in d_pos]))
        for d_ner in self.d_ner:
            self.d_ner_tensor.append(torch.LongTensor([ner_vocab[w] for w in d_ner]))

        self.q_pos_tensor = torch.LongTensor([pos_vocab[w] for w in self.q_pos])
        self.q_features = torch.from_numpy(self.q_features).type(torch.FloatTensor)
        
        self.p_q_relation = []
        self.p_c_relation = []
        for p_q_relation in input_dict['p_q_relations']:
            self.p_q_relation.append(torch.LongTensor([rel_vocab[r] for r in p_q_relation]))

        for p_c_relation in input_dict['p_c_relations']:
            self.p_c_relation.append(torch.LongTensor([rel_vocab[r] for r in p_c_relation]))

            
    def __str__(self):
        print_string = []
        for passage, choice in zip(self.passages, self.choices):
            print_string.append('Passage: %s\n Question: %s\n Answer: %s, Label: %d\n' % (passage, self.question, choice, self.label))
        return "".join(print_string)


def _to_indices_and_mask_elmo(batch_text, elmo):
    # character_ids = batch_to_ids(batch_text).cuda() # remember to transfer to gpu
    # embeddings = elmo(character_ids)    
    # representations = embeddings['elmo_representations'][0] # depends on elmo_num_layer
    # mask = embeddings['mask'] != 1 # hacky way to flip 0 and 1 to keep consistent with trian
    # return representations, mask.byte() 
    
    activations, mask = elmo.batch_to_embeddings(batch_text)
    activations = activations.sum(axis=1)
    mask = mask != 1
    return activations, mask

def _to_indices_and_mask(batch_tensor, need_mask=True):
    mx_len = max([t.size(0) for t in batch_tensor])
    batch_size = len(batch_tensor)
    indices = torch.LongTensor(batch_size, mx_len).fill_(0)
    if need_mask:
        mask = torch.ByteTensor(batch_size, mx_len).fill_(1)
    for i, t in enumerate(batch_tensor):
        indices[i, :len(t)].copy_(t)
        if need_mask:
            mask[i, :len(t)].fill_(0)
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

                                 
def _to_feature_tensor_flatten(features):
    mx_len = 0
    for sample in features:
        for f in sample:
            mx_len = max(mx_len, f.shape[0])
    
    batch_size = len(features)
    f_dim = features[0][0].shape[1]
    
    f_tensor = torch.FloatTensor(batch_size, 4, mx_len, f_dim).fill_(0)
    
    for i, sample in enumerate(features):
        for j, f in enumerate(sample):
            f_tensor[i, j, :len(f), :].copy_(torch.from_numpy(f))
    
    return f_tensor
                                 
                                 
def flatten_tensor(tensor, need_mask=True):
    batch_size = len(tensor)
    flatten = [] # flatten
    for t in tensor:
        flatten.extend(t)
    if need_mask:
        t, t_mask = _to_indices_and_mask(flatten)
        t = t.reshape(batch_size, 4, t.size(-1)) # B*K x T -> B x K x T
        t_mask = t_mask.reshape(batch_size, 4, t_mask.size(-1)) # B*K x T -> B x K x T
        return t,t_mask
    else:
        t = _to_indices_and_mask(flatten, need_mask)
        t = t.reshape(batch_size, 4, t.size(-1)) # B*K x T -> B x K x T
        return t


def batchify(batch_data, elmo=None):
    batch_size = len(batch_data)
    
    # tensor related with p and c are list of tensors. in other words, in the shape of B x K x T                             
    p_pos = flatten_tensor([ex.d_pos_tensor for ex in batch_data], need_mask=False)
    p_ner = flatten_tensor([ex.d_ner_tensor for ex in batch_data], need_mask=False)
    p_q_relation = flatten_tensor([ex.p_q_relation for ex in batch_data], need_mask=False)
    p_c_relation = flatten_tensor([ex.p_c_relation for ex in batch_data], need_mask=False) 
    
    q_pos = _to_indices_and_mask([ex.q_pos_tensor for ex in batch_data], need_mask=False)
    q_f_tensor = _to_feature_tensor([ex.q_features for ex in batch_data]) # B x T

    f_tensor = _to_feature_tensor_flatten([ex.features for ex in batch_data])

    y = torch.FloatTensor([ex.label for ex in batch_data])
    
    if elmo==None:
        p, p_mask = flatten_tensor([ex.d_tensor for ex in batch_data])
        q, q_mask = _to_indices_and_mask([ex.q_tensor for ex in batch_data])
        c, c_mask = flatten_tensor([ex.c_tensor for ex in batch_data])

    return p, p_pos, p_ner, p_mask, q, q_pos, q_mask, c, c_mask, f_tensor, q_f_tensor, p_q_relation, p_c_relation, y 

