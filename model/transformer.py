import torch
import torch.nn as nn


# Subsequence Mask
def get_subsequence_mask(seq_len, pred_len, device ,pt=0):
    if pt == 0:
        subsequence_mask = torch.zeros((seq_len, seq_len)).to(device)
        subsequence_mask[:, -pred_len:] = float("-inf")
    elif pt == 1:
        # Upper triangular matrix
        subsequence_mask = torch.triu(torch.full((seq_len, seq_len), float("-inf")), diagonal=1).to(device)
    else:
        raise RuntimeError(f"Not such subsequence mask decode_mode:{pt}.")

    return subsequence_mask


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_head):
        super(ScaledDotProductAttention, self).__init__()
        self.d_head = d_head
    def forward(self, Q, K, V, mask):
        """
        Q: [batch_size, n_heads, len_q, d_head)]
        K: [batch_size, n_heads, len_k(=len_v), d_head]
        V: [batch_size, n_heads, len_v(=len_k), d_head]
        mask: [batch_size, n_heads, seq_len, seq_len]
        """
        scores = torch.matmul(Q, K.transpose(-1, -2)) / torch.sqrt(
            torch.tensor(self.d_head, dtype=torch.float32))  # scores : [batch_size, n_heads, len_q, len_k]

        if mask is not None:
            scores += mask

        attn = torch.softmax(scores, dim=-1)
        # attn: [batch_size, n_heads, len_q, len_k]
        context = torch.matmul(attn, V)
        # context: [batch_size, n_heads, len_q, d_head]
        return context


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, d_head):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_head
        self.W_Q = nn.Linear(self.d_model, self.d_model, bias=False)
        self.W_K = nn.Linear(self.d_model, self.d_model, bias=False)
        self.W_V = nn.Linear(self.d_model, self.d_model, bias=False)
        self.fc = nn.Linear(self.d_model, self.d_model, bias=False)

    def forward(self, input_Q, input_K, input_V, mask):
        """
        input_Q: [batch_size, len_q, d_model]
        input_K: [batch_size, len_k(=len_v), d_model]
        input_V: [batch_size, len_v(=len_k), d_model]
        attn_mask: [batch_size, seq_len, seq_len]
        """
        batch_size = input_Q.size(0)
        Q = self.W_Q(input_Q).view(batch_size, -1, self.n_heads, self.d_head).transpose(1, 2)
        K = self.W_K(input_K).view(batch_size, -1, self.n_heads, self.d_head).transpose(1, 2)
        V = self.W_V(input_V).view(batch_size, -1, self.n_heads, self.d_head).transpose(1, 2)

        context = ScaledDotProductAttention(self.d_head)(Q, K, V, mask)
        # context: [batch_size, n_heads, len_q, d_head]
        context = context.transpose(1, 2).reshape(batch_size, -1, self.n_heads * self.d_head)
        output = self.fc(context)
        # output: [batch_size, len_q, d_model]
        return output


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(FeedForward, self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )

    def forward(self, inputs):
        """
        inputs: [batch_size, seq_len, d_model]
        """
        output = self.feed_forward(inputs)
        return output

    
# EncoderLayer
class EncoderLayer(nn.Module):
    def __init__(self, d_model, dropout_rate, n_heads, d_head, d_ff):
        super(EncoderLayer, self).__init__()
        self.d_model = d_model
        self.dropout_rate = dropout_rate
        self.n_heads = n_heads
        self.d_head = d_head
        self.d_ff = d_ff
        self.enc_self_attn = MultiHeadAttention(d_model=self.d_model, n_heads=self.n_heads, d_head=self.d_head)
        self.ffn = FeedForward(d_model=self.d_model, d_ff=self.d_ff)
        self.norm1 = nn.LayerNorm(self.d_model)
        self.norm2 = nn.LayerNorm(self.d_model)
        self.dropout = nn.Dropout(self.dropout_rate)

    def forward(self, enc_layer_inputs, enc_self_attn_mask):
        """
        enc_layer_inputs: [batch_size, src_len, d_model]
        enc_self_attn_mask: [batch_size, src_len, src_len]
        """
        # enc_layer_inputs (Q, K, V is the same)
        residual1 = enc_layer_inputs.clone()
        enc_self_attn_outputs = self.enc_self_attn(enc_layer_inputs, enc_layer_inputs, enc_layer_inputs,
                                                   enc_self_attn_mask)
        outputs1 = self.norm1(enc_self_attn_outputs + residual1)

        residual2 = outputs1.clone()
        ffn_outputs = self.ffn(outputs1)
        # ffn_outputs: [batch_size, src_len, d_model]
        ffn_outputs = self.dropout(ffn_outputs)
        outputs2 = self.norm2(ffn_outputs + residual2)

        return outputs2


# # EncoderLayer

# Encoder
class Encoder(nn.Module):
    def __init__(self, n_encoder_layers, d_model, dropout_rate, n_heads, d_head, d_ff):
        super(Encoder, self).__init__()
        self.n_encoder_layers = n_encoder_layers
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_head
        self.d_ff = d_ff
        self.dropout_rate = dropout_rate

        self.layers = nn.ModuleList([EncoderLayer(d_model=self.d_model, dropout_rate=self.dropout_rate, \
                                                  n_heads=self.n_heads, d_head = self.d_head, d_ff = self.d_ff) for _ in range(n_encoder_layers)])

    def forward(self, enc_inputs, enc_self_attn_mask):
        """
        enc_inputs: [batch_size, src_len]
        """
        enc_outputs = enc_inputs.clone()
        for layer in self.layers:
            enc_outputs = layer(enc_outputs, enc_self_attn_mask)
        return enc_outputs



# DecoderLayer
class DecoderLayer(nn.Module):
    def __init__(self, d_model, dropout_rate, n_heads, d_head, d_ff):
        super(DecoderLayer, self).__init__()
        self.d_model = d_model
        self.dropout_rate = dropout_rate
        self.n_heads = n_heads
        self.d_head = d_head
        self.d_ff = d_ff

        self.dec_self_attn = MultiHeadAttention(self.d_model, self.n_heads, self.d_head)
        self.norm1 = nn.LayerNorm(self.d_model)
        self.dec_enc_attn = MultiHeadAttention(self.d_model, self.n_heads, self.d_head)
        self.norm2 = nn.LayerNorm(self.d_model)
        self.ffn = FeedForward(self.d_model, self.d_ff)
        self.norm3 = nn.LayerNorm(self.d_model)
        self.dropout = nn.Dropout(self.dropout_rate)

    def forward(self, dec_layer_inputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask):
        """
        dec_layer_inputs: [batch_size, tgt_len, d_model]
        enc_outputs: [batch_size, src_len, d_model]
        dec_self_attn_mask: [batch_size, tgt_len, tgt_len]
        dec_enc_attn_mask: [batch_size, tgt_len, src_len]
        """
        residual1 = dec_layer_inputs.clone()
        dec_self_attn_outputs = self.dec_self_attn(dec_layer_inputs, dec_layer_inputs, dec_layer_inputs,
                                                   dec_self_attn_mask)
        outputs1 = self.norm1(dec_self_attn_outputs + residual1)

        residual2 = outputs1.clone()
        dec_enc_attn_outputs = self.dec_enc_attn(outputs1, enc_outputs, enc_outputs, dec_enc_attn_mask)
        outputs2 = self.norm2(dec_enc_attn_outputs + residual2)

        residual3 = outputs2.clone()
        ffn_outputs = self.ffn(outputs2)
        ffn_outputs = self.dropout(ffn_outputs)
        outputs3 = self.norm3(ffn_outputs + residual3)

        return outputs3


# class DecoderLayer(nn.Module):
#     def __init__(self, d_model, dropout_rate, n_heads, d_head, d_ff):

# Decoder
class Decoder(nn.Module):
    def __init__(self, n_decoder_layers, d_model, dropout_rate\
                 , n_heads, d_head, d_ff):
        super(Decoder, self).__init__()
   
        self.n_heads = n_heads
        self.d_head = d_head
        self.d_ff = d_ff
        self.n_decoder_layers = n_decoder_layers
        self.d_model = d_model
        self.dropout_rate = dropout_rate
        self.layers = nn.ModuleList([DecoderLayer(self.d_model, self.dropout_rate, n_heads=self.n_heads, \
                                                  d_head=self.d_head, d_ff = self.d_ff) for _ in range(self.n_decoder_layers)])

    def forward(self, dec_inputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask):
        """
        dec_inputs: [batch_size, tgt_len, d_model]
        enc_intpus: [batch_size, src_len, d_model]
        enc_outputs: [batsh_size, src_len, d_model]
        """
        dec_outputs = dec_inputs.clone()
        for layer in self.layers:
            dec_outputs = layer(dec_outputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask)
        return dec_outputs


class Transformer(nn.Module):
    def __init__(self, src_len, src_size, tgt_size, dropout_rate, d_model, 
                 n_heads, n_encoder_layers, n_decoder_layers, d_ff, d_head, pred_len, device,
                 ):
        super(Transformer, self).__init__()
        self.seq_len = src_len 
        self.num_layers = n_encoder_layers
        self.dropout_rate = dropout_rate
        self.src_size = src_size
        self.tgt_size = tgt_size
        self.src_len = src_len
        self.d_model = d_model
        self.pred_len = pred_len
        self.device = device
        self.n_heads = n_heads
        self.n_encoder_layers = n_encoder_layers
        self.n_decoder_layers = n_decoder_layers
        self.d_ff = d_ff
        self.d_head = d_head
        self.dropout = nn.Dropout(dropout_rate)
        self.encoder = Encoder(n_encoder_layers=self.n_encoder_layers, d_model=self.d_model, \
                               dropout_rate=self.dropout_rate, n_heads=self.n_heads, d_head=self.d_head, d_ff=self.d_ff)
        self.decoder = Decoder(n_decoder_layers=self.n_decoder_layers, d_model=self.d_model, \
                               dropout_rate=self.dropout_rate, n_heads=self.n_heads, d_head=self.d_head, d_ff=self.d_ff)

        
        self.tgt_linear = nn.Linear(self.tgt_size, self.d_model)
        self.src_linear = nn.Linear(self.src_size, self.d_model)
        self.linear_project = nn.Linear(self.d_model, self.tgt_size)


    def positional_encoding(self, seq_len, d_model):
        import math
        PE = torch.zeros(seq_len, d_model).to("cuda:0")
        for pos in range(seq_len):
            for i in range(0, d_model, 2):
                PE[pos, i] = math.sin(pos / (10000 ** ((2 * i) / d_model)))
                if i + 1 < d_model:
                    PE[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))
        return PE
    

    def forward(self, src, tgt):
        # Position Embedding and Input Projection
        batch_size = src.shape[0]
        src_position = self.positional_encoding(self.seq_len, self.d_model)
        tgt_position = self.positional_encoding(self.src_len, self.d_model) # expected shape: (src_len, d_model)
        src_inputs = src_position.unsqueeze(0).repeat(batch_size, 1, 1) + self.src_linear(src)
        tgt_inputs = tgt_position.unsqueeze(0).repeat(batch_size, 1, 1) + self.tgt_linear(tgt) # tgt_lin_output # expected shape: (batch_size, tgt_len, d_model)

        # Encoder
        enc_self_attn_mask = None
        enc_outputs = self.encoder(src_inputs, enc_self_attn_mask)
        # enc_outputs: [batch_size, src_len, d_model]

        # Decoder
        # dec_self_attn_mask = get_subsequence_mask(self.src_len, self.pred_len, self.device, pt=0)  # dec_self_attn_mask: [tgt_len, tgt_len]
        dec_self_attn_mask = get_subsequence_mask(self.src_len,self.pred_len,self.device, pt=0)
        dec_enc_attn_mask = None
        dec_outputs = self.decoder(tgt_inputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask)
        # dec_outputs: [batch_size, tgt_len, d_model]

        # Linear project
        project_outputs = self.linear_project(dec_outputs)
        # project_outputs: [batch_size, tgt_len, tgt_size]

        return project_outputs
    
    def get_attention_score(self, src):
        batch_size = src.shape[0]
        src_position = self.positional_encoding(self.src_len, self.d_model)
        src_inputs = src_position.unsqueeze(0).repeat(batch_size, 1, 1) + self.embedding(src)
        enc_self_attn_mask = None
        enc_outputs = self.encoder(src_inputs, enc_self_attn_mask)
        return enc_outputs
    
    def get_embdding_parameters(self):
        return self.src_linear.parameters()