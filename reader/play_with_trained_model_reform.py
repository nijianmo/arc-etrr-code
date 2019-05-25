from config import args
from utils import load_data, build_vocab, gen_submission, gen_final_submission, eval_based_on_outputs
from model import Model

if __name__ == '__main__':
    if not args.pretrained:
        print('No pretrained model specified.')
        exit(0)
    build_vocab()
    # swap dev and test
    swap = False
    if swap:
        args.test_mode = False if args.test_mode else True

    if args.test_mode:
        dev_path = './data/ARC-Challenge-Test-question-reform.nn-qa-para.clean-processed.json'
        print("load test data...")
    else:
        dev_path = './data/ARC-Challenge-Dev-question-reform.nn-qa-para.clean-processed.json'
        print("load dev data...")
        
    dev_data = load_data(dev_path)

    # ensemble models
    model_path_list = args.pretrained.split(',')
    for model_path in model_path_list:
        print('Load model from %s...' % model_path)
        args.pretrained = model_path
        model = Model(args)

        # evaluate on development dataset
        dev_acc = model.evaluate(dev_data, debug=True)
        #dev_acc = model.evaluate(dev_data)
        print('dev accuracy: %f' % dev_acc)

