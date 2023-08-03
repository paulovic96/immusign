import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class NonLinearModel(torch.nn.Module):
    def __init__(self,
                 input_channel,
                 output_channel,
                 hidden_units,
                 activation_function=torch.nn.LeakyReLU(),
                 global_average_pooling = True):
        
        super().__init__()
        self.activation_function = activation_function
        self.non_linear = torch.nn.Linear(input_channel, hidden_units)
        self.linear = torch.nn.Linear(hidden_units, output_channel)
        self.global_average_pooling = global_average_pooling

    def forward(self, x, *args):
        output = self.non_linear(x)
        output = self.activation_function(output)     
        output = self.linear(output)
        if self.global_average_pooling and len(output.shape) == 3:
            output = output.mean([1])    
        return output


class ResnetModel(torch.nn.Module):
    def __init__(self,
                 input_channel,
                 output_channel,
                 hidden_units,
                 hidden_layers,
                 activation_function=torch.nn.LeakyReLU(),
                 global_average_pooling = True):
        super().__init__()
        self.input_channel = input_channel
        self.hidden_units = hidden_units
        self.activation_function = activation_function

        self.resid_layers = torch.nn.ModuleList(
            [torch.nn.Linear(input_channel, hidden_units)])
        self.resid_layers = self.resid_layers.extend(torch.nn.ModuleList(
            [torch.nn.Linear(hidden_units, hidden_units) for _ in range(hidden_layers - 1)]))
        self.output_layer = torch.nn.Linear(hidden_units, output_channel)
        self.global_average_pooling = global_average_pooling

    def forward(self, x, *args):
        output = x
        for i, block in enumerate(self.resid_layers):
            resid = output
            output = block(output)
            output = self.activation_function(output)
            if i == 0:
                if self.input_channel == self.hidden_units:
                    output += resid
            else:
                output += resid

        output = self.output_layer(output)
        if self.global_average_pooling and len(output.shape) == 3:
            output = output.mean([1])  
        return output
    

class LSTMNet(torch.nn.Module):
    def __init__(self,
                 input_dim, 
                 output_dim, 
                 lstm_layers,
                 lstm_units,
                 bidirectional,
                 dropout,
                 fc_layer,
                 fc_hidden_dim,
                activation_function=torch.nn.LeakyReLU(),
                 batch_first = True):
        
        super(LSTMNet,self).__init__()

        
        
        self.num_directions = 2 if bidirectional else 1
        self.bidirectional = bidirectional
        self.activation_function = activation_function
        self.fc_layer = fc_layer
        self.dropout = dropout
        
        self.lstm = torch.nn.LSTM(input_dim,
                            lstm_units,
                            num_layers = lstm_layers,
                            bidirectional = bidirectional,
                            dropout = dropout,
                            batch_first = batch_first
                           )
        
        # Dense layer to predict 
        self.fc1 = torch.nn.Linear(self.num_directions * lstm_units, fc_hidden_dim)
        if dropout:
            self.dropout_layer = torch.nn.Dropout(dropout)
        if fc_layer > 1:
            self.fc_n = torch.nn.ModuleList(
                [torch.nn.Linear(fc_hidden_dim, fc_hidden_dim) for _ in range(fc_layer - 1)])

        self.output_layer = torch.nn.Linear(fc_hidden_dim, output_dim)

    
    def forward(self,seq,seq_len):
        
        packed_embedded = pack_padded_sequence(seq, seq_len, batch_first=True, enforce_sorted = False)
        output, (h_n, c_n) = self.lstm(packed_embedded)
        output_unpacked, output_lengths = pad_packed_sequence(output, batch_first=True)
        output = output_unpacked[:, -1, :]
        output = self.activation_function(output)
        output = self.fc1(output)
        if self.dropout:
            output = self.dropout_layer(output)
        if self.fc_layer > 1:
            for i, fc_layer in enumerate(self.fc_n):
                output = self.activation_function(output)
                output = fc_layer(output)
                if self.dropout:
                    output = self.dropout_layer(output)
                
        output = self.output_layer(output)
        return output