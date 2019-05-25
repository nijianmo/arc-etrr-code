import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import *
import arc_solvers.tests.cnn as cnn

    
'''
    select phrases from a query that should be efficient to retrieve documents from seaerch engine
'''    
class Generator(nn.Module):

    def __init__(self, embeddings, args):
        super(Generator, self).__init__()
        vocab_size, hidden_dim = embeddings.shape
        self.embedding_layer = nn.Embedding( vocab_size, hidden_dim)
        self.embedding_layer.weight.data = torch.from_numpy( embeddings )
        self.embedding_layer.weight.requires_grad = False
        self.args = args
        if args.model_form == 'cnn':
            self.cnn = cnn.CNN(args, max_pool_over_time = False)

        self.z_dim = 2

        self.hidden = nn.Linear((len(args.filters)* args.filter_num), self.z_dim)
        self.dropout = nn.Dropout(args.dropout)
        
    def _z_forward(self, activ): # (Batch, Embed, Length) / (Batch * num_choice, Embed, Length)
        '''
            Returns prob of each token being selected
        '''
        activ = activ.transpose(1,2) # turn back to (Batch, Length, Embed) 
        activ = activ.reshape(self.batch_size, self.num_choice, activ.size(1), activ.size(2)) # (Batch, num_choice, Length, Embed)
        logits = self.hidden(activ) # (Batch, num_choice, Length, z_dim)
        probs = F.relu(F.softmax(logits, dim=3)) # output prob
        # probs = F.softmax(F.relu(logits), dim=2) # softmax over words of one choice
        z = probs[:,:,:,1]
        return z # (Batch, Length) / (Batch, num_choice, Length)

    def forward(self, x_indx):
        '''
            Given input x_indx of dim (batch, 1, num_choice, length), return z (batch, num_choice, length) such that z can act as element-wise mask on x
        '''
        if self.args.model_form == 'cnn':
            x = self.embedding_layer(x_indx.squeeze(1)) # squeeze 
            if self.args.cuda:
                x = x.cuda()
                
            self.batch_size = x.size(0)
            self.num_choice = x.size(1)
            x = x.reshape(x.size(0) * x.size(1), x.size(2), x.size(3)) # (Batch * num_choice, Length, Embed)    
            x = torch.transpose(x, 1, 2) # Switch X to (Batch, Embed, Length)
            activ = self.cnn(x) # (Batch, Embed, Length) 
        else:
            raise NotImplementedError("Model form {} not yet supported for generator!".format(args.model_form))

        z = self._z_forward(F.relu(activ)) # probability
        selection = self.sample(z) # final selection
        return selection, z 


    def sample(self, z):
        '''
            Get selection from probablites at each token. 
            Sample at train time, hard selection at test time
        '''
        if self.training:
            # sample
            selection = torch.bernoulli(z)
        else:
            ## pointwise set <thres to 0 >=thres to 1
            thres = 0.5
            selection = z > thres
        return selection.float() # make sure not byte tensor

