#############################################
#  Author: Hongwei Fan                      #
#  E-mail: hwnorm@outlook.com               #
#  Homepage: https://github.com/hwfan       #
#############################################
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Training Activity2Vec')
    parser.add_argument('--cfg',
            default='', 
            type=str,
            help='define the configuration file of model.')
    parser.add_argument('--model',
            default='', 
            type=str,
            help='define the name of model.')
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
    parser.add_argument('--cfg',
            default='', 
            type=str,
            help='define the configuration file of model.')
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