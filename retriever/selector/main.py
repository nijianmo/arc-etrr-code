import os
import time
import torch
import random
import numpy as np

from datetime import datetime

from utils import load_data, build_vocab
from config import args
from model import Model

torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

if __name__ == '__main__':
    
    build_vocab()

    model = Model(args)      
    train_data = load_data('./data/train-terms-processed.json', model.elmo)
    dev_data = load_data('./data/dev-terms-processed.json', model.elmo)
    if args.test_mode:
        # use validation data as training data
        train_data += dev_data
        dev_data = []

    # best_dev_acc = 0.0
    best_dev_f1 = 0.0 # compare f1
    
    os.makedirs('./checkpoint', exist_ok=True)
    checkpoint_path = './checkpoint/%d-%s.mdl' % (args.seed, datetime.now().isoformat())
    print('Trained model will be saved to %s' % checkpoint_path)

    for i in range(args.epoch):
        print('Epoch %d...' % i)
        if i == 0:
            dev_acc, dev_precision, dev_recall, dev_f1 = model.evaluate(dev_data)
            print('Dev accuracy={:.3f}, precision={:.3f}, recall={:.3f}, f1={:.3f}'.format(dev_acc, dev_precision, dev_recall, dev_f1))
        start_time = time.time()
        np.random.shuffle(train_data)
        cur_train_data = train_data

        model.train(cur_train_data)
        train_acc, train_precision, train_recall, train_f1 = model.evaluate(train_data[:2000], debug=False, eval_train=True)
        print('Train accuracy={:.3f}, precision={:.3f}, recall={:.3f}, f1={:.3f}'.format(train_acc, train_precision, train_recall, train_f1))
        
        dev_acc, dev_precision, dev_recall, dev_f1 = model.evaluate(dev_data, debug=True)
        print('Dev accuracy={:.3f}, precision={:.3f}, recall={:.3f}, f1={:.3f}'.format(dev_acc, dev_precision, dev_recall, dev_f1))

        # if dev_acc > best_dev_acc:
        # best_dev_acc = dev_acc
        if dev_f1 > best_dev_f1:
            best_dev_f1 = dev_f1
            model.save(checkpoint_path)
        elif args.test_mode:
            model.save(checkpoint_path)
        print('Epoch %d use %d seconds.' % (i, time.time() - start_time))

    # print('Best dev accuracy: %f' % best_dev_acc)
    print('Best dev f1={:.3f}'.format(best_dev_f1))
