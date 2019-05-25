import os
import json
import torch
from config import args
from utils import load_data, build_vocab
from model import Model

if __name__ == '__main__':
    if not args.pretrained:
        print('No pretrained model specified.')
        exit(0)
    build_vocab()

    model_path = args.pretrained
    print('Load model from %s...' % model_path)
    model = Model(args)

    # evaluate 
    model.network.eval()
    batch_size = args.batch_size

    for partition in ['Train','Dev','Test']:
        terms = {}
        data = load_data('./data/{}-'.format(args.dataset) + partition + '-processed.json', model.elmo) 

        for batch_num, batch_input in enumerate(model._iter_data(data)):
            feed_input = [x for x in batch_input[:-3]]  
            q_mask = feed_input[3]
            q_is_cand, y, y_mask = batch_input[-3], batch_input[-2], batch_input[-1] # now y has mask 

            pred_proba = model.network(*feed_input)
            length = torch.sum(q_mask.eq(0), dim=1)

            #pred = pred_proba.argmax(dim=-1) # predict
            pred = pred_proba.squeeze(2) > 0.5 

            for idx in range(len(y)):
                l = length[idx]
                term = []

                idx2 = batch_num * batch_size + idx
                d = data[idx2]
                text = d.question.split()

                for i,w in zip(pred[idx][:l], text):
                    if i == 1:
                        term.append(w)

                terms[idx2] = term

        ### update reform json
        path = '../arc-solver/data/ARC-V1-Feb2018/{}'.format(args.dataset)
        fdev = os.path.join(path, '{}-'.format(args.dataset) + partition + '.jsonl')
        
        json_dev = []
        with open(fdev, 'r') as f:
            for l in f.readlines():
                json_dev.append(json.loads(l.strip()))
        print(len(json_dev))
        print(len(terms))

        for idx in range(len(data)):
            term = terms[idx]
            json_dev[idx]['question_reform'] = " ".join(term)

        # write to disk
        fdev = os.path.join(path, '{}-'.format(args.dataset) + partition + '-question-reform.nn.jsonl')
        with open(fdev, 'w') as f:
            for l in json_dev:
                f.write(json.dumps(l) + '\n')

