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
#
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

# def change_tree(tree):
#     for child in tree.children:
#         child.value = tree.value+5
#         change_tree(child)
# lambda_tree.value = 5
# change_tree(lambda_tree)
# ptree(lambda_tree)

print(for_tree.value)
class LambdaGrammar(IntEnum):
    INT = 0
    VAR_NAME = 1
    VAR = 2
    EXPR = 3
    VARAPP = 4
    CMP = 5
    TERM = 6
    VARUNIT = 7


class Lambda(IntEnum):
    VAR = 0
    CONST = 1
    PLUS = 2
    MINUS = 3
    EQUAL = 4
    LE = 5
    GE = 6
    IF = 7
    LET = 8
    UNIT = 9
    LETREC = 10
    APP = 11
    ROOT = 12

lambda_grammar = {
        Lambda.ROOT: [LambdaGrammar.TERM],
        Lambda.VAR: [LambdaGrammar.VAR_NAME],
        Lambda.CONST: [LambdaGrammar.INT],
        Lambda.PLUS: [LambdaGrammar.EXPR, LambdaGrammar.EXPR],
        Lambda.MINUS: [LambdaGrammar.EXPR, LambdaGrammar.EXPR],
        Lambda.EQUAL: [LambdaGrammar.EXPR, LambdaGrammar.EXPR],
        Lambda.LE: [LambdaGrammar.EXPR, LambdaGrammar.EXPR],
        Lambda.GE: [LambdaGrammar.EXPR, LambdaGrammar.EXPR],
        Lambda.IF: [LambdaGrammar.CMP, LambdaGrammar.TERM, LambdaGrammar.TERM],
        Lambda.LET: [LambdaGrammar.VARUNIT, LambdaGrammar.TERM, LambdaGrammar.TERM],
        Lambda.UNIT: [],
        Lambda.LETREC: [LambdaGrammar.VAR_NAME, LambdaGrammar.VAR_NAME, LambdaGrammar.TERM,
                        LambdaGrammar.TERM],
        Lambda.APP: [LambdaGrammar.VARAPP, LambdaGrammar.EXPR]
    }

def category_to_child_LAMBDA(num_vars, num_ints, category):
    """
    Take a category of output, and return a list of new tokens which can be its children in the
    Lambda language.

    :param num_vars: number of variables a program can use
    :param num_ints: number of ints a program can use
    :param category: category of output generated next
    """
    n = num_ints + num_vars
    lambda_grammar = {
        LambdaGrammar.INT: range(num_ints),
        LambdaGrammar.VAR_NAME: range(num_ints, n),
        LambdaGrammar.VAR: [x + n for x in [Lambda.VAR]],
        LambdaGrammar.EXPR: [x + n for x in [Lambda.VAR, Lambda.CONST, Lambda.PLUS, Lambda.MINUS, Lambda.CONST]],
        LambdaGrammar.VARAPP: [x + n for x in [Lambda.VAR, Lambda.APP]] + list(range(num_ints, n)),
        LambdaGrammar.CMP: [x + n for x in [Lambda.EQUAL, Lambda.LE, Lambda.GE]],
        LambdaGrammar.TERM: [x + n for x in [Lambda.LET, Lambda.LETREC, Lambda.PLUS, Lambda.MINUS, Lambda.VAR,
                                             Lambda.CONST, Lambda.UNIT, Lambda.IF, Lambda.APP]],
        LambdaGrammar.VARUNIT: [x + n for x in [Lambda.VAR]] + list(range(num_ints, n)),
    }

    return lambda_grammar[category]