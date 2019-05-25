import numpy as np
import random
import time
import torch

import arc_solvers.tests.utils as utils
from collections import OrderedDict
from typing import Dict, List
from arc_solvers.tests.es_search import EsSearch, EsHit
from arc_solvers.tests.utils import MAX_HITS

def get_search_metric(selection, xi_dict, xi_oov, yi, es_search, query_mode):
    '''
    convert binary selection into queries, search via ES, get answer, and compare with truth
    '''
    # N and T
    num_choice, max_len = selection.shape

    xi = xi_dict['x']
    mask_lens = [min(len(d), max_len) for d in xi] # mask_len cannot exceed max_len
    mask = build_mask(max_len, mask_lens)

    if query_mode == "hypothesis":  # reformulate the hypothesis
        # new query
        queries = []
        num_choice  = selection.size(0)
        for j in range(num_choice):
            query = []
            for k,v in enumerate(selection[j][:mask_lens[j]]): # the kth word in the original query
                if v != 0 or xi_oov[j][k] != 0: # keep it either because it's selected or it's oov
                    query.append(xi[j][k])
            # combine terms and append jth query
            queries.append(" ".join(query))

    elif query_mode == "question":  # only reformulate the question 
        choices = xi_dict['choices_tok']
 
        # new query
        queries = []
        num_choice  = selection.size(0)
        for j in range(num_choice):
            query = []
            for k,v in enumerate(selection[j][:mask_lens[j]]): # the kth word in the original query
                if v != 0 or xi_oov[j][k] != 0: # keep it either because it's selected or it's oov
                    query.append(xi[j][k])
            query.extend(choices[j]) # concatenate the answer
            # combine terms and append jth query
            queries.append(" ".join(query))

    # search queries
    hits_per_choice = es_search.get_hits_for_question_rl(queries)
        
    # pick the hits with topk score
    filter_hits_across_choices(hits_per_choice, MAX_HITS)
    
    # pick hits with high score
    ir_scores = []
    for choice, hits in hits_per_choice.items():
        if len(hits) > 0:
            hit = hits[0]
            ir_scores.append(hit.score)
        else: # if no hit remains, consider as 0 score
            ir_scores.append(0)
    ans = np.argmax(np.array(ir_scores))
    # convert into paras
    
    # return
    if (ans == yi):
        reward = 1.0
    else:
        reward = 0.0
        
    return reward, mask, queries # return reward and mask tensor

       
def build_mask(max_len, mask_lens):
    mask = []
    for mask_len in mask_lens:
        l = [1] * mask_len
        l.extend([0] * (max_len - mask_len))
        mask.append(l)
    return torch.ByteTensor(mask)
    
    
def filter_hits_across_choices(hits_per_choice: Dict[str, List[EsHit]],
                               top_k: int):
    """
    Filter the hits from all answer choices(in-place) to the top_k hits based on the hit score
    """
    # collect ir scores
    ir_scores = [hit.score for hits in hits_per_choice.values() for hit in hits]
    # if more than top_k hits were found
    if len(ir_scores) > top_k:
        # find the score of the top_kth hit
        min_score = sorted(ir_scores, reverse=True)[top_k - 1]
        # filter hits below this score
        for choice, hits in hits_per_choice.items():
            hits[:] = [hit for hit in hits if hit.score >= min_score]    


