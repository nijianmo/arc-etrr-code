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

    if args.test_mode:
        dev_data = load_data('./data/test-terms-processed.json', model.elmo)
    else:
        dev_data = load_data('./data/dev-terms-processed.json', model.elmo)

    # evaluate on development dataset
    dev_acc, dev_precision, dev_recall, dev_f1 = model.evaluate(dev_data)
    print('Dev accuracy={:.3f}, precision={:.3f}, recall={:.3f}, f1={:.3f}'.format(dev_acc, dev_precision, dev_recall, dev_f1))

