import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--image_size',choices=[7,14,28] ,type=int,default=14)
parser.add_argument('--layer_num',type=int,choices=[1,2,3,4],default=4)
parser.add_argument('--train_epoch',type=int,default=20)
parser.add_argument('--hidden_layer_size',type=int,default=14)
parser.add_argument('--distill_epoch',type=int,default=5)
parser.add_argument('--prune_amount',type=float,default=0.5)
args = parser.parse_args()
