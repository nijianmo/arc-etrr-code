import torch
import torch.utils.data as data
import gzip
import arc_solvers.tests.utils as utils
import tqdm
import numpy as np
import pickle
import json
from arc_solvers.tests.datasets.factory import RegisterDataset
from arc_solvers.tests.datasets.abstract_dataset import AbstractDataset
import os

from arc_solvers.tests.utils import MAX_LEN

SMALL_TRAIN_SIZE = 800
LABEL_MAP = {'A':1, '1':1, 
             'B':2, '2':2,
             'C':3, '3':3,
             'D':4, '4':4}

@RegisterDataset('arc')
class ArcDataset(AbstractDataset):

    ### mode: train/valid/test
    ### query_mode: hypothesis/question
    def __init__(self, args, word_to_indx, mode, query_mode='hypothesis', max_length=MAX_LEN, stem='/data2/jianmo/reading_comprehension/ARC-V1-Feb2018/ARC-Challenge/raw_with_hypothesis'):
        self.args= args
        self.name = mode
        self.dataset = []
        self.text = {}
        self.word_to_indx  = word_to_indx
        self.max_length = max_length
        self.name_to_key = {'train':'train', 'dev':'heldout', 'test':'heldout'}
        self.query_mode = query_mode

        if mode == 'train':
            file_name = os.path.join(stem, "ARC-Challenge-Train.clean.jsonl") # "ARC-Challenge-Train.jsonl"
        elif mode == 'dev':
            file_name = os.path.join(stem, "ARC-Challenge-Dev.clean.jsonl")
        elif mode == 'test':
            file_name = os.path.join(stem, "ARC-Challenge-Test.clean.jsonl")
            
        with open(file_name, 'r') as f:
            lines = f.readlines()

            for indx, line in tqdm.tqdm(enumerate(lines)):
                line_content = json.loads(line.strip())
                # sample = self.processLine(line_content, indx)
                sample, raw = self.processLine(line_content, indx)

                self.dataset.append(sample)
                self.text[raw[0]] = raw[1] # raw[0] is index, raw[1] is text dict
            
    def flatten(self, sents):
        return [word for sent in sents for word in sent] # flatten double list
    
    
    ## Convert one line from arc dataset to a sample
    def processLine(self, line, i):
        
        try:
            y = LABEL_MAP[line['answerKey']]
        except:
            print(line)
            raise ValueError('bad answer key...')
        
        choices = line['question']['choices']
        question_tok = line['question']['question_tok']      
        question_tok = self.flatten(question_tok)

        # reformulate hypothesis
        if self.query_mode == "hypothesis": 
            x = [] # size is (num_choice x Length)
            choices_tok = [] 
            for choice in choices:
                hypothesis_tok = choice['hypothesis_tok']
                hypothesis_tok = self.flatten(hypothesis_tok)
                choice_tok = choice['choice_tok']
                choice_tok = self.flatten(choice_tok)
                x.append(hypothesis_tok)
                choices_tok.append(choice_tok)       
 
            # max length?!
            x_indx, x_oov = utils.get_indices_tensor_rl(x, self.word_to_indx, self.max_length)
        
            sample = {'y':y, 'i':i, 'x_indx':x_indx, 'x_oov':x_oov}
            raw = [i, {'question_tok':question_tok, 'x':x, 'choices_tok':choices_tok}]

        # reformulate question
        elif self.query_mode == "question":
            x = [question_tok] * len(choices)
            choices_tok = []
            for choice in choices:
                choice_tok = choice['choice_tok']
                choice_tok = self.flatten(choice_tok)
                choices_tok.append(choice_tok)

            # max length?!
            x_indx, x_oov = utils.get_indices_tensor_rl(x, self.word_to_indx, self.max_length)

            sample = {'y':y, 'i':i, 'x_indx':x_indx, 'x_oov':x_oov}
            raw = [i, {'question_tok':question_tok, 'x':x, 'choices_tok':choices_tok}]

        # return
        return sample, raw    
