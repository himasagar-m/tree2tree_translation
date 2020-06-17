import torch
import torch.nn as nn
from tree_to_sequence.translating_trees import (Node, pretty_print_tree, Lambda)


class TreeToTreeAttention(nn.Module):
    def __init__(self, encoder, decoder, hidden_size, embedding_size, nclass,
                 root_value=-1, alignment_size=50, align_type=1, max_size=50):
        """
        Translates an encoded representation of one tree into another
        """
        super(TreeToTreeAttention, self).__init__()

        # Save useful values
        self.nclass = nclass
        self.encoder = encoder
        self.decoder = decoder
        self.align_type = align_type
        self.max_size = max_size
        self.root_value = root_value

        # EOS is always the last token
        self.EOS_value = nclass

        # Useful functions
        self.softmax = nn.Softmax(0)
        self.tanh = nn.Tanh()

        # Set up attention
        if align_type == 0:
            self.attention_hidden = nn.Linear(hidden_size, alignment_size)
            self.attention_context = nn.Linear(hidden_size, alignment_size, bias=False)
            self.attention_alignment_vector = nn.Linear(alignment_size, 1)
        elif align_type == 1:
            self.attention_hidden = nn.Linear(hidden_size, hidden_size)

        self.attention_presoftmax = nn.Linear(2 * hidden_size, hidden_size)
        self.embedding = nn.Embedding(nclass + 1, embedding_size)
        self.relation_linear_layer = nn.Linear(hidden_size * 4, hidden_size)

    def forward_train(self, input_tree, target_tree, teacher_forcing=True):
        """
        Generate predictions for an output tree given an input tree, then calculate the loss.
        """
        # Encode tree

        annotations, decoder_hiddens, decoder_cell_states = self.encoder(input_tree)

        if self.align_type <= 1:
            attention_hidden_values = self.attention_hidden(annotations)
        else:
            attention_hidden_values = annotations

        # number to accumulate loss
        loss = 0

        # Tuple: (hidden_state, cell_state, desired_output, parent_value,parent_ec, child_index)
        unexpanded = [(decoder_hiddens, decoder_cell_states, target_tree, self.root_value,True,None, 0)]

        # while stack isn't empty:
        while (len(unexpanded)) > 0:
            # Pop last item
            decoder_hiddens, decoder_cell_states, targetNode, parent_val,parent_ec,parent_rel, child_index = \
                unexpanded.pop()

            # Use attention and past hidden state to generate scores
            decoder_hidden = decoder_hiddens[-1].unsqueeze(0)
            attention_logits = self.attention_logits(attention_hidden_values, decoder_hidden)
            attention_probs = self.softmax(attention_logits)  # number_of_nodes x 1
            context_vec = (attention_probs * annotations).sum(0).unsqueeze(0)  # 1 x hidden_size
            et = self.tanh(self.attention_presoftmax(torch.cat((decoder_hiddens, context_vec),
                                                               dim=1)))  # 1 x hidden_size
            # Calculate loss
            loss += self.decoder.calculate_loss(parent_val,targetNode.ec,targetNode.rel, child_index, et, targetNode.value)

            # If we have an EOS, there are no children to generate
            if int(targetNode.value) == self.EOS_value:
                continue

            # Teacher forcing means we use the correct value (not the predicted value) of a node to 
            # generate its children
            next_ec = targetNode.ec
            next_rel = targetNode.relation
            if teacher_forcing:
                next_input = targetNode.value
                next_ec = targetNode.ec
                next_rel = targetNode.relation

            else:
                _, next_input = log_odds.topk(1)

            decoder_input = torch.cat((self.embedding(next_input), et), 1)

            for i, child in enumerate(targetNode.children):
                # Parent node of a node's children is that node
                parent = next_input
                new_child_index = i

                # Get hidden state and cell state which will be used to generate this node's 
                # children
                child_hiddens, child_cell_states = self.decoder.get_next(parent, new_child_index,
                                                                         decoder_input,
                                                                         decoder_hiddens,
                                                                         decoder_cell_states)
                unexpanded.append([child_hiddens, child_cell_states, child, parent, next_ec,next_rel,
                                   new_child_index])

        return loss
    def find_relation(self, parent_states, child_states, attention_hidden_values, annotations, parent, child_index,return_loss=False):
        # relation prediction between nodes

        # here -- relace attention_hidden_values with relation_hidden_values
        decoder_hiddens = self.relation_linear_layer(torch.cat((parent_states, child_states), 0))
        attention_logits = self.attention_logits(attention_hidden_values, decoder_hiddens)
        attention_probs = self.softmax(attention_logits)  # number_of_nodes x 1
        context_vec = (attention_probs * annotations).sum(0).unsqueeze(0)  # 1 x hidden_size
        et = self.tanh(self.attention_presoftmax(torch.cat((decoder_hiddens, context_vec),
                                                           dim=1)))  # 1 x hidden_size
        log_odds, possible_indices = self.get_log_odds(parent, child_index, et)
        _, max_index = torch.max(log_odds, 1)
        predicted_relation = torch.tensor([possible_indices[int(max_index)]], device=max_index.device)
        if return_loss == False:
            return predicted_relation
        else :
            loss = self.loss_func(log_odds, predicted_relation)
    def find_last_child(self,decoder_hiddens, decoder_cell_states, curr_root, parent_val,parent_ec,parent_rel, child_index):


        return ec
    def forward_prediction(self, input_tree, max_size=None):
        """
        Generate an output tree given an input tree
        """
        if max_size is None:
            max_size = self.max_size

        # Encode tree
        annotations, decoder_hiddens, decoder_cell_states = self.encoder(input_tree)

        # Counter of how many nodes we've generated so far
        num_nodes = 0

        PLACEHOLDER = 1

        # Output tree we're building up.  The value is a placeholder
        tree = Node(PLACEHOLDER)

        # Create stack of unexpanded nodes
        # Tuple: (hidden_state, cell_state, desired_output, parent_value, child_index)
        #unexpanded = [(decoder_hiddens, decoder_cell_states, tree, self.root_value,None, 0)]
        unexpanded = [(decoder_hiddens, decoder_cell_states, tree, self.root_value,True,None, 0)]

        if self.align_type <= 1:
            attention_hidden_values = self.attention_hidden(annotations)
        else:
            attention_hidden_values = annotations

        # while stack isn't empty:
        while (len(unexpanded)) > 0:
            # Pop last item
            decoder_hiddens, decoder_cell_states, curr_root, parent_val,parent_ec,parent_rel, child_index = \
                unexpanded.pop()



            # Use attention and pass hidden state to make a prediction
            decoder_hiddens = decoder_hiddens[-1].unsqueeze(0)
            attention_logits = self.attention_logits(attention_hidden_values, decoder_hiddens)
            attention_probs = self.softmax(attention_logits)  # number_of_nodes x 1
            context_vec = (attention_probs * annotations).sum(0).unsqueeze(0)  # 1 x hidden_size
            et = self.tanh(self.attention_presoftmax(torch.cat((decoder_hiddens, context_vec),
                                                               dim=1)))  # 1 x hidden_size
            next_input,ec,rel = self.decoder.make_prediction(parent_val, et)
            curr_root.value = next_input
            curr_root.ec = ec
            curr_root.relation = rel
            num_nodes += 1

            # Only generate up to max_size nodes
            if num_nodes > self.max_size :
                break

            # If we have an EOS, there are no children to generate
            if int(curr_root.value) == self.EOS_value :
                continue
            embed = torch.cat()
            decoder_input = torch.cat((self.embedding(next_input), et), 1)
            parent = next_input

            for i in range(20):
                # Get hidden state and cell state which will be used to generate this node's 
                # children
                child_hiddens, child_cell_states = self.decoder.get_next(parent, i,
                                                                         decoder_input,
                                                                         decoder_hiddens,
                                                                         decoder_cell_states)
                # Add children to the stack
                curr_child = Node(PLACEHOLDER)
                # parent_states = torch.cat((decoder_hiddens, decoder_cell_states), 0)
                # child_states = torch.cat((child_hiddens, child_cell_states), 0)
                #parent_relation = self.find_relation(parent_states, child_states, attention_hidden_values, annotations, parent, i, return_loss=False)
                # unexpanded = [(decoder_hiddens, decoder_cell_states, tree, self.root_value,True,None, 0)]
                unexpanded.append((child_hiddens, child_cell_states, curr_child, parent,curr_root.ec,curr_root.relation, i))
                curr_root.children.append(curr_child)

                decoder_hiddens, decoder_cell_states, curr_root, parent_val,parent_ec,parent_rel, child_index  \
                    = child_hiddens, child_cell_states, curr_child, parent,curr_root.ec,curr_root.relation, i
                decoder_hiddens = decoder_hiddens[-1].unsqueeze(0)
                attention_logits = self.attention_logits(attention_hidden_values, decoder_hiddens)
                attention_probs = self.softmax(attention_logits)  # number_of_nodes x 1
                context_vec = (attention_probs * annotations).sum(0).unsqueeze(0)  # 1 x hidden_size
                et = self.tanh(self.attention_presoftmax(torch.cat((decoder_hiddens, context_vec),
                                                                   dim=1)))  # 1 x hidden_size
                next_input,ec,rel = self.decoder.make_prediction(parent_val, et)

                if ec == 89:
                    break

        return tree


    def print_example(self, input_tree, target_tree):
        """
        Print out the desired and actual output trees for one example
        """
        print("DESIRED")
        pretty_print_tree(target_tree)
        print("WE GOT!")
        pretty_print_tree(self.forward_prediction(input_tree))

    def attention_logits(self, attention_hidden_values, decoder_hidden):
        """
        Calculates the logits over the nodes in the input tree.
        """
        if self.align_type == 0:
            return self.attention_alignment_vector(self.tanh(self.attention_context(decoder_hidden)
                                                             + attention_hidden_values))
        else:
            return (decoder_hidden * attention_hidden_values).sum(1).unsqueeze(1)

    def update_max_size(self, max_size):
        self.max_size = max_size
