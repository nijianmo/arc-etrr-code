import torch
import torch.nn as nn
import torch.nn.functional as F

import layers
from utils import vocab, pos_vocab, ner_vocab, rel_vocab

class TriAN(nn.Module):

    def __init__(self, args):
        super(TriAN, self).__init__()
        self.args = args

        if self.args.use_elmo:
            self.embedding_dim = self.args.elmo_num_layer * 1024 
        else:
            self.embedding_dim = 300
            self.embedding = nn.Embedding(len(vocab), self.embedding_dim, padding_idx=0) # len is same as vocab size
            self.embedding.weight.data.fill_(0)
            self.embedding.weight.data[:2].normal_(0, 0.1) # initialize

        self.pos_embedding = nn.Embedding(len(pos_vocab), args.pos_emb_dim, padding_idx=0)
        self.pos_embedding.weight.data.normal_(0, 0.1)
        self.ner_embedding = nn.Embedding(len(ner_vocab), args.ner_emb_dim, padding_idx=0)
        self.ner_embedding.weight.data.normal_(0, 0.1)
        self.rel_embedding = nn.Embedding(len(rel_vocab), args.rel_emb_dim, padding_idx=0)
        self.rel_embedding.weight.data.normal_(0, 0.1)
        self.RNN_TYPES = {'lstm': nn.LSTM, 'gru': nn.GRU}

        self.p_q_emb_match = layers.SeqAttnMatch(self.embedding_dim) # question-aware passage representation
        self.c_q_emb_match = layers.SeqAttnMatch(self.embedding_dim) # question-aware choice representation
        self.c_p_emb_match = layers.SeqAttnMatch(self.embedding_dim) # passage-aware choice representation

        # Input size to RNN: word emb + question emb + pos emb + ner emb + manual features
        doc_input_size = 2 * self.embedding_dim + args.pos_emb_dim + args.ner_emb_dim + 5 + 2 * args.rel_emb_dim

        # RNN document encoder
        self.doc_rnn = layers.StackedBRNN(
            input_size=doc_input_size,
            hidden_size=args.hidden_size,
            num_layers=args.doc_layers,
            dropout_rate=0,
            dropout_output=args.dropout_rnn_output,
            concat_layers=False,
            rnn_type=self.RNN_TYPES[args.rnn_type],
            padding=args.rnn_padding)

        # RNN question encoder: word emb + pos emb
        qst_input_size = self.embedding_dim + args.pos_emb_dim
        self.question_rnn = layers.StackedBRNN(
            input_size=qst_input_size,
            hidden_size=args.hidden_size,
            num_layers=1,
            dropout_rate=0,
            dropout_output=args.dropout_rnn_output,
            concat_layers=False,
            rnn_type=self.RNN_TYPES[args.rnn_type],
            padding=args.rnn_padding)

        # RNN answer encoder
        choice_input_size = 3 * self.embedding_dim
        self.choice_rnn = layers.StackedBRNN(
            input_size=choice_input_size,
            hidden_size=args.hidden_size,
            num_layers=1,
            dropout_rate=0,
            dropout_output=args.dropout_rnn_output,
            concat_layers=False,
            rnn_type=self.RNN_TYPES[args.rnn_type],
            padding=args.rnn_padding)

        # Output sizes of rnn encoders
        doc_hidden_size = 2 * args.hidden_size
        question_hidden_size = 2 * args.hidden_size
        choice_hidden_size = 2 * args.hidden_size

        # Answer merging
        self.c_self_attn = layers.LinearSeqAttn(choice_hidden_size)
        self.q_self_attn = layers.LinearSeqAttn(question_hidden_size + 1) # add essential term flag

        self.c_diff_attn = layers.DiffSeqAttn(choice_hidden_size)

        self.p_q_attn = layers.BilinearSeqAttn(x_size=doc_hidden_size, y_size=question_hidden_size)

        #self.p_c_bilinear = nn.Linear(doc_hidden_size, choice_hidden_size)
        #self.q_c_bilinear = nn.Linear(question_hidden_size, choice_hidden_size)
        self.p_c_bilinear = nn.Linear(2*doc_hidden_size, 3*choice_hidden_size)
        self.q_c_bilinear = nn.Linear(2*question_hidden_size, 3*choice_hidden_size)

    def expand_dim(self, tensor):
        tensor = tensor.unsqueeze(1)
        tensor = tensor.repeat(1,4,1,1) # add choice dim
        return tensor
    
        
    def forward(self, p, p_pos, p_ner, p_mask, q, q_pos, q_mask, c, c_mask, f_tensor, q_f_tensor, p_q_relation, p_c_relation):
        '''
        shape:
        p: B x 4 x T
        p_pos: B x 4 x T
        p_ner: B x 4 x T
        p_mask: B x 4 x T
        q: B x T
        q_pos: B x T
        c: B x 4 x T
        c_mask: B x 4 x T
        f_tensor: B x 4 x T x dim
        q_f_tensor: B x 4 x dim(=1)
        p_q_relation: B x 4 x T
        p_c_relation: B x 4 x T
        '''
        
        batch_size = p.size(0)
        T = p.size(2)
        
        p_emb = self.embedding(p)
        p_pos_emb = self.pos_embedding(p_pos)
        p_ner_emb = self.ner_embedding(p_ner)
        p_q_rel_emb = self.rel_embedding(p_q_relation)
        
        q_emb = self.embedding(q)
        q_pos_emb = self.pos_embedding(q_pos)

        # Dropout on embeddings
        if self.args.dropout_emb > 0:
            p_emb = nn.functional.dropout(p_emb, p=self.args.dropout_emb, training=self.training)
            p_pos_emb = nn.functional.dropout(p_pos_emb, p=self.args.dropout_emb, training=self.training)
            p_ner_emb = nn.functional.dropout(p_ner_emb, p=self.args.dropout_emb, training=self.training)
            p_q_rel_emb = nn.functional.dropout(p_q_rel_emb, p=self.args.dropout_emb, training=self.training)
            
            q_emb = nn.functional.dropout(q_emb, p=self.args.dropout_emb, training=self.training)
            q_pos_emb = nn.functional.dropout(q_pos_emb, p=self.args.dropout_emb, training=self.training)
       
        # q rnn 
        q_rnn_input = torch.cat([q_emb, q_pos_emb], dim=2)
        q_hiddens = self.question_rnn(q_rnn_input, q_mask)
        # final q
        q_merge_weights = self.q_self_attn(torch.cat([q_hiddens, q_f_tensor], dim=2), q_mask) # BxT2
        q_hidden = layers.weighted_avg(q_hiddens, q_merge_weights) # BxH
        
        q_hidden_max, _ = torch.max(q_hiddens, dim=1)
        final_q_hidden = torch.cat([q_hidden_max, q_hidden], dim=1)
 
        p_hiddens, c_hiddens = [], []
        # iterate all choices
        for i in range(4):
            pi_emb = p_emb[:,i,:,:].contiguous()
            pi_pos_emb = p_pos_emb[:,i,:,:].contiguous()
            pi_ner_emb = p_ner_emb[:,i,:,:].contiguous()
            pi_q_rel_emb = p_q_rel_emb[:,i,:,:].contiguous()
            pi_mask = p_mask[:,i,:].contiguous()
            
            ci = c[:,i,:].contiguous()
            ci_mask = c_mask[:,i,:].contiguous()
            pi_ci_relation = p_c_relation[:,i,:].contiguous()
        
            ci_emb = self.embedding(ci)
            pi_ci_rel_emb = self.rel_embedding(pi_ci_relation)

            ci_emb = nn.functional.dropout(ci_emb, p=self.args.dropout_emb, training=self.training)
            pi_ci_rel_emb = nn.functional.dropout(pi_ci_rel_emb, p=self.args.dropout_emb, training=self.training)
            pi_q_weighted_emb = self.p_q_emb_match(pi_emb, q_emb, q_mask)
            pi_q_weighted_emb = nn.functional.dropout(pi_q_weighted_emb, p=self.args.dropout_emb, training=self.training)        
            ci_q_weighted_emb = self.c_q_emb_match(ci_emb, q_emb, q_mask)
            ci_pi_weighted_emb = self.c_p_emb_match(ci_emb, pi_emb, pi_mask)

            
            ci_q_weighted_emb = nn.functional.dropout(ci_q_weighted_emb, p=self.args.dropout_emb, training=self.training)
            ci_pi_weighted_emb = nn.functional.dropout(ci_pi_weighted_emb, p=self.args.dropout_emb, training=self.training)
        
            # concat
            fi_tensor = f_tensor[:,i,:,:].contiguous()
            pi_rnn_input = torch.cat([pi_emb, pi_q_weighted_emb, pi_pos_emb, pi_ner_emb, fi_tensor, pi_q_rel_emb, pi_ci_rel_emb], dim=2)
            ci_rnn_input = torch.cat([ci_emb, ci_q_weighted_emb, ci_pi_weighted_emb], dim=2)      

            # pass RNN
            pi_hiddens = self.doc_rnn(pi_rnn_input, pi_mask) # BxT1xH
            ci_hiddens = self.choice_rnn(ci_rnn_input, ci_mask) # BxT2xH

            p_hiddens.append(pi_hiddens)
            c_hiddens.append(ci_hiddens)

        # calculate score
        probs = []
        pc_scores, qc_scores = [], []
        # iterate all choices
        for i in range(4):
            pi_hiddens = p_hiddens[i]
            ci_hiddens = c_hiddens[i]
            pi_mask = p_mask[:,i,:].contiguous()
            ci_mask = c_mask[:,i,:].contiguous() 

            # final p
            pi_merge_weights = self.p_q_attn(pi_hiddens, q_hidden, pi_mask)
            pi_hidden = layers.weighted_avg(pi_hiddens, pi_merge_weights)

            # final c
            ci_merge_weights = self.c_self_attn(ci_hiddens, ci_mask)
            # add diff net between choices
            ci_diff_weights = self.c_diff_attn(c_hiddens, i, ci_mask)   

            ci_hidden1 = layers.weighted_avg(ci_hiddens, ci_merge_weights)
            ci_hidden2 = layers.weighted_avg(ci_hiddens, ci_diff_weights)
            ci_hidden_max, _ = torch.max(ci_hiddens, dim=1)       

            #pc_score = torch.sum(self.p_c_bilinear(pi_hidden) * ci_hidden, dim=-1) # B x h x B x h -> B x 1
            #qc_score = torch.sum(self.q_c_bilinear(q_hidden) * ci_hidden, dim=-1) # B x h x B x h -> B x 1

            #pc_score = torch.sum(self.p_c_bilinear(pi_hidden) * ci_hidden1, dim=-1) # B x h x B x h -> B x 1
            #qc_score = torch.sum(self.q_c_bilinear(q_hidden) * ci_hidden2, dim=-1) # B x h x B x h -> B x 1

            #final_ci_hidden = torch.cat([ci_hidden1, ci_hidden2], dim=1)
            final_ci_hidden = torch.cat([ci_hidden1, ci_hidden2, ci_hidden_max], dim=1)
            
            pi_hidden_max, _ = torch.max(pi_hiddens, dim=1)       
            final_pi_hidden = torch.cat([pi_hidden_max, pi_hidden], dim=1)

            pc_score = torch.sum(self.p_c_bilinear(final_pi_hidden) * final_ci_hidden, dim=-1) # B x h x B x h -> B x 1
            qc_score = torch.sum(self.q_c_bilinear(final_q_hidden) * final_ci_hidden, dim=-1) # B x h x B x h -> B x 1

            pc_scores.append(pc_score)
            qc_scores.append(qc_score)

        pc_scores = torch.stack(pc_scores, dim=-1)
        qc_scores = torch.stack(qc_scores, dim=-1)
        pc_scores= pc_scores.reshape(batch_size, 4) # B x 4
        qc_scores= qc_scores.reshape(batch_size, 4) # B x 4
        pc_scores = F.softmax(pc_scores, dim=-1)
        qc_scores = F.softmax(qc_scores, dim=-1)
 
        probs = (pc_scores + qc_scores)
        return probs

