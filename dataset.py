import argparse

from tree_to_sequence.program_datasets import *
from tree_to_sequence.tree_encoder import TreeEncoder

parser = argparse.ArgumentParser()

parser.add_argument('--save_folder', default="test_various_models",
                    help='Name of folder to save files in. Defaults to test_various_models/')
parser.add_argument('--cuda_device', type=int, default=0,
                    help='Number of cuda device. Not relevant if cuda is disabled. Default is 0.')
parser.add_argument('--num_vars', type=int, default=10, help='Number of variable names. Default is 10.')
parser.add_argument('--num_ints', type=int, default=11, help='Number of possible integer literals. Default is 11')
parser.add_argument('--one_hot', action='store_true', help='Use one hot vectors instead of embeddings.')
parser.add_argument('--binarize_input', action='store_true', help="Binarize the input. Default is not to.")
parser.add_argument('--binarize_output', action='store_true', help="Binarize the output. Default is not to.")
parser.add_argument('--binary_tree_lstm_cell', action='store_true',
                    help="Use a binary tree lstm cell. Default is not to.")
parser.add_argument('--no_long_base_case', action='store_true',
                    help="Use a more minimal tree (mainly dropping out tokens that don't add any information)")
parser.add_argument('--lr', type=float, default=0.005, help='learning rate for model using adam, default=0.005')
parser.add_argument('--dropout', type=float, default=False,
                    help='Dropout probability. The default is not to use dropout.')
parser.add_argument('--num_epochs', type=int, default=5, help='Number of epochs to train for. The default is 5.')
parser.add_argument('--no_cuda', action='store_true', help='Disables cuda')
opt = parser.parse_args()

save_folder = opt.save_folder

num_vars = opt.num_vars
eos_token = False
num_ints = opt.num_ints
one_hot = opt.one_hot
binarize_input = opt.binarize_input
binarize_output = opt.binarize_output
long_base_case = not opt.no_long_base_case
input_as_seq = False
output_as_seq = False
# dset_training = ForLambdaDataset("training_For.json",
#                                        binarize_input=binarize_input, binarize_output=binarize_output,
#                                        eos_token=eos_token, one_hot=one_hot,
#                                        num_ints=num_ints, num_vars=num_vars,
#                                        long_base_case=long_base_case,
#                                        input_as_seq=input_as_seq,
#                                        output_as_seq=output_as_seq)
#
#
# print(dir(dset_training))
# print(type(dset_training))
#
# # import pickle
# # with open('test_for','wb') as data_test_for :
# #     pickle.dump(dset_test,data_test_for)
# torch.save(dset_training,'dset_training_for')
path = 'training_For.json'
progs_json = json.load(open(path))

for_tree = make_tree_for(progs_json[0], long_base_case=long_base_case)
lambda_tree = translate_from_for(copy.deepcopy(for_tree))


def ptree(root):
    print("-"*20)
    print(root.value)
    print("*"*20)
    if len(root.children) > 0:
        for c in root.children:
            print(c.value)
    print("-"*20)
    if len(root.children) > 0:
        for c in root.children:
            ptree(c)

def change_tree(tree):
    for child in tree.children:
        child.value = tree.value+5
        change_tree(child)
lambda_tree.value = 5
change_tree(lambda_tree)
#ptree(lambda_tree)

encoder = TreeEncoder(10, 10, 1, [1, 2, 3, 4, 5], attention=True, one_hot=False,
                          binary_tree_lstm_cell=False)