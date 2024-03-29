import torch
import torch.nn as nn


class EncoderRNN1(nn.Module):
    def __init__(self, seq_len_e1, output_len_e1, input_size_e1, hidden_size_e1):
        super().__init__()
        self.seq_len_e1 = seq_len_e1
        self.output_len_e1 = output_len_e1
        self.input_size_e1 = input_size_e1
        self.hidden_size_e1 = hidden_size_e1

        self.lstm = nn.LSTM(input_size=self.input_size_e1, hidden_size=self.hidden_size_e1, num_layers=1, bias=True,
                            dropout=0, batch_first=True, bidirectional=False)

    def forward(self, inputs):
        output, (h_n, c_n) = self.lstm(inputs)

        return output[:, -self.output_len_e1:, :]


class EncoderRNN2(nn.Module):
    def __init__(self, seq_len_e2, input_size_e2, hidden_size_e2):
        super().__init__()
        self.seq_len_e2 = seq_len_e2
        self.input_size_e2 = input_size_e2
        self.hidden_size_e2 = hidden_size_e2

        self.lstm = nn.LSTM(input_size=self.input_size_e2, hidden_size=self.hidden_size_e2, num_layers=1, bias=True,
                            dropout=0, batch_first=True, bidirectional=False)

    def forward(self, inputs):
        output, _ = self.lstm(inputs)

        return output[:, [-1], :]


class DecoderLSTM(nn.Module):
    def __init__(self, output_len_d, input_size_d, hidden_size_d, dropout_rate):
        super().__init__()
        self.output_len_d = output_len_d
        self.input_size_d = input_size_d
        self.hidden_size_d = hidden_size_d
        self.dropout_rate = dropout_rate

        self.dropout = nn.Dropout(p=self.dropout_rate)
        self.lstm = nn.LSTM(input_size=self.input_size_d, hidden_size=self.hidden_size_d, num_layers=1, bias=True,
                            dropout=0, batch_first=True, bidirectional=False)

        self.out1 = nn.Linear(in_features=self.hidden_size_d, out_features=1, bias=True)

    def forward(self, inputs):
        output, _ = self.lstm(inputs)
        output = self.dropout(output)
        final_output = self.out1(output)

        return final_output


class LSTMMSVS2S(nn.Module):
    
    def __init__(self,src_len,pred_len,src_size,hidden_size,\
                 past_len,tgt_size,\
                dropout_rate):
        super().__init__()
        self.encoder_obs = EncoderRNN1(src_len, pred_len, src_size, hidden_size)
        self.encoder_runoff = EncoderRNN2(past_len, tgt_size, hidden_size)
        self.decoder = DecoderLSTM(pred_len, hidden_size*2, hidden_size, dropout_rate)
        self.pred_len = pred_len

    def forward(self, seq_x, seq_y_past):
        # seq_x : (batch_size, seq_len, input_size)
        # seq_y_past : (batch_size, past_len, input_size)
        encoder_obs_outputs = self.encoder_obs(seq_x)
        encoder_runoff_outputs = self.encoder_runoff(seq_y_past)
        encoder_runoff_outputs = encoder_runoff_outputs.expand(-1, self.pred_len, -1)

        decoder_inputs = torch.cat((encoder_obs_outputs, encoder_runoff_outputs), dim=2)
        output = self.decoder(decoder_inputs)

        return output
