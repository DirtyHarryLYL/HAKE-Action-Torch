import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Training Activity2Vec')
    parser.add_argument('--model',
            default='', 
            type=str,
            help='define the name of model.')
    parser.add_argument('--resume', 
            action='store_true',
            help='choose whether to resume from pretrained model.')
    parser.add_argument('--trained-modules', 
            default='foot,leg,hip,hand,arm,head,verb', 
            type=str,
            help='choose the modules to train.')
    parser.add_argument('--data-splits', 
            default='', 
            type=str,
            help='choose the data splits for training. Using all HAKE-Large data as default.')
    parser.add_argument(
        "opts",
        help="See activity2vec/ult/config.py for all options",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()
    return args

def test_parse_args():
    parser = argparse.ArgumentParser(description='Testing Activity2Vec')
    parser.add_argument('--eval', 
            help='specific stage for evaluation',
            default=1,
            type=int)
    parser.add_argument('--benchmark', 
            help='specific stage for evaluation',
            default=1,
            type=int)
    parser.add_argument(
        "opts",
        help="See activity2vec/ult/config.py for all options",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()
    return args