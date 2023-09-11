import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import torch.nn.functional as F
import scipy.sparse as sp
from itertools import repeat
from sklearn.base import BaseEstimator, ClassifierMixin
from transformer_modules import TransformerBlock, ExtractionModule
from sys import platform


def gen(src, index, dim=-1, out=None, dim_size=None, fill_value=0):
    dim = range(src.dim())[dim]  # Get real dim value.

    # Automatically expand index tensor to the right dimensions.
    if index.dim() == 1:
        index_size = list(repeat(1, src.dim()))
        index_size[dim] = src.size(dim)
        index = index.view(index_size).expand_as(src)

    # Generate output tensor if not given.
    if out is None:
        dim_size = index.max().item() + 1 if dim_size is None else dim_size
        out_size = list(src.size())
        out_size[dim] = dim_size
        out = src.new_full(out_size, fill_value)

    return src, out, index, dim

def global_add_pool(x, batch, size=None):
    size = batch[-1].item() + 1 if size is None else size
    return scatter_add(x, batch, dim=0, dim_size=size)

def scatter_add(src, index, dim=-1, out=None, dim_size=None, fill_value=0):
    src, out, index, dim = gen(src, index, dim, out, dim_size, fill_value)
    return out.scatter_add_(dim, index, src)

class ScaledDotProductAttentionLayer(nn.Module):
    def __init__(self, input_dim):
        super(ScaledDotProductAttentionLayer, self).__init__()
        self.scaling_factor = torch.rsqrt(torch.tensor(input_dim, dtype=torch.float32))

    def compute_weights(self, x, batch_index):   
        unique_repertoires = batch_index.unique()
        scores = []
        for repertoire in unique_repertoires:
            mask = (batch_index == repertoire).unsqueeze(1)
            x_repertoire = x[mask.expand_as(x)].view(-1, x.size(1))

            # Compute scaled dot-product attention scores
            attention_scores = torch.matmul(x_repertoire, x_repertoire.t()) * self.scaling_factor
            attention_weights = F.softmax(attention_scores, dim=-1)
            scores.append(attention_weights)

        return scores
    
    def forward(self, x, batch_index):
        unique_repertoires = batch_index.unique()
        attended_features_list = []

        for repertoire in unique_repertoires:
            mask = (batch_index == repertoire).unsqueeze(1)
            x_repertoire = x[mask.expand_as(x)].view(-1, x.size(1))

            # Compute scaled dot-product attention scores
            attention_scores = torch.matmul(x_repertoire, x_repertoire.t()) * self.scaling_factor
            attention_weights = F.softmax(attention_scores, dim=-1)

            attended_features = torch.matmul(attention_weights, x_repertoire)
            attended_features_list.append(attended_features.mean(dim=0))

        attended_features_tensor = torch.stack(attended_features_list, dim=0)

        return attended_features_tensor
    
class AttentionLayer(nn.Module):
    def __init__(self, input_dim, attention_dim, output_dim):
        super(AttentionLayer, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, output_dim)
        )


    def compute_weights(self, x, batch_index):   
        unique_repertoires = batch_index.unique()
        scores = []
        for repertoire in unique_repertoires:
            mask = (batch_index == repertoire).unsqueeze(1)
            x_repertoire = x[mask.expand_as(x)].view(-1, x.size(1))
            attention_scores = self.attention(x_repertoire)
            attention_weights = torch.softmax(attention_scores, dim=0)
            scores.append(attention_weights)
        return scores
      

    def forward(self, x, batch_index):
        # x: Tensor with shape (batch_size * num_clones, feature_dim)
        # batch_index: Tensor with shape (batch_size * num_clones)
        # This tensor indicates which repertoire each clone belongs to (0 for repertoire 0, 1 for repertoire 1, and so on)

        unique_repertoires = batch_index.unique()
        attended_features_list = []

        for repertoire in unique_repertoires:
            mask = (batch_index == repertoire).unsqueeze(1)
            x_repertoire = x[mask.expand_as(x)].view(-1, x.size(1))
            attention_scores = self.attention(x_repertoire)
            attention_weights = torch.softmax(attention_scores, dim=0)
            attended_features = torch.sum(attention_weights * x_repertoire, dim=0)
            attended_features_list.append(attended_features)

        attended_features_tensor = torch.stack(attended_features_list, dim=0)
        return attended_features_tensor

class CrossAttentionLayer(nn.Module):
    def __init__(self, input_dim, output_dim, num_heads=4, num_extraction_layers=1, num_interaction_layers=1):
        super(CrossAttentionLayer, self).__init__()
        attention_dim = output_dim
        
        self.target_emb = nn.Parameter(torch.randn(1, 1, attention_dim))

        self.interaction_blocks = nn.ModuleList([TransformerBlock(attention_dim, num_heads)
                                                 for _ in range(num_interaction_layers)])
        self.extraction_blocks = nn.ModuleList([ExtractionModule(attention_dim, num_heads)
                                                for _ in range(num_extraction_layers)])
      

    def compute_weights(self, x, batch_index):   
        attention = self.forward(x, batch_index, return_attention=True)
        return attention
      

    def forward(self, x, batch_index, return_attention = False):
        # x: Tensor with shape (batch_size * num_clones, feature_dim)
        # batch_index: Tensor with shape (batch_size * num_clones)
        # This tensor indicates which repertoire each clone belongs to (0 for repertoire 0, 1 for repertoire 1, and so on)

        unique_repertoires = batch_index.unique()
        attended_features_list = []
        attentions = []
        for repertoire in unique_repertoires:
            mask = (batch_index == repertoire).unsqueeze(1)
            x_repertoire = x[mask.expand_as(x)].view(-1, x.size(1)).unsqueeze(0)
            target_token = self.target_emb.expand(1, -1, -1)

            for interaction_block in self.interaction_blocks:
               target_token, x_repertoire = interaction_block(target_token, x_repertoire)

            for extraction_block in self.extraction_blocks:
                target_token, attn = extraction_block(target_token, x_repertoire, return_attention=True)

            attn = attn.squeeze(0)
            attn = attn.mean(dim=1).squeeze(1)
            attended_features_list.append(target_token.squeeze(0))
            attentions.append(attn)

        attended_features_tensor = torch.stack(attended_features_list, dim=0).squeeze(1)
        if return_attention:
            return attentions
        return attended_features_tensor
    

class Phi(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, dropout=None):
        super(Phi, self).__init__()
        self.fc_in = nn.Linear(input_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.num_layers = num_layers
        self.dropout = dropout
        if self.num_layers > 2: # 3 layers
            self.fc_hidden = nn.Linear(hidden_dim, hidden_dim)

        if self.dropout is not None:
            self.dropout = nn.Dropout(dropout)


    def forward(self, x):
        x = F.relu(self.fc_in(x))
        if self.dropout is not None:
            x = self.dropout(x)
        if self.num_layers > 2:
            x = F.relu(self.fc_hidden(x))
        x = self.fc_out(x)
        return x

# Define the DeepSet network
class DeepSet(nn.Module):
    def __init__(self, var_input_dim, batch_size=2, phi_hidden_dim=128, 
                 phi_output_dim=128, rho_hidden_dim=128, rho_output_dim=128,
                 attention_dim=64, attention_type='scaled_dot_product', dropout=None):
        super(DeepSet, self).__init__()
        self.var_input_dim = var_input_dim
        
        self.batch_size = batch_size
        
        self.phi = Phi(self.var_input_dim, phi_hidden_dim, phi_output_dim)
        self.rho_input_dim = phi_output_dim 
        self.rho = Phi(self.rho_input_dim, rho_hidden_dim, rho_output_dim, 3, dropout=dropout)
        
        if attention_type == 'scaled_dot_product':
            self.attention_layer = ScaledDotProductAttentionLayer(phi_output_dim)
        elif attention_type == 'cross_attention':
            self.attention_layer = CrossAttentionLayer(phi_output_dim, phi_output_dim)
        else:
            self.attention_layer = AttentionLayer(phi_output_dim, attention_dim, phi_output_dim)
        
       
        if platform == "darwin":
            use_gpu = torch.backends.mps.is_available()
            device = "mps" if torch.backends.mps.is_available() else "cpu"  
        else:
            use_gpu = torch.cuda.is_available()
            device = "cuda" if torch.cuda.is_available() else "cpu"

            self.attention_layer = self.attention_layer.to(device)
            self.phi = self.phi.to(device)
            self.rho = self.rho.to(device)
        self.use_gpu = use_gpu
        self.device = device

    def compute_attention(self, x):
        var_states = []
        batch_index = []
        j = 0
        for i, s in enumerate(x):
            var_state = torch.tensor(s.reshape(-1, self.var_input_dim))
            var_states.extend(var_state)
            batch_index.extend(torch.repeat_interleave(torch.tensor(j), len(var_state)))  
            j += 1
        # concat all var_states
        var_states_var = torch.stack(var_states)

        if self.use_gpu:
            var_states_var = var_states_var.to(self.device, dtype=torch.float32)

        x = self.phi(var_states_var) 
        batch_index = torch.LongTensor(batch_index)
        if self.use_gpu:
            batch_index = batch_index.to(self.device)
 

        attention_scores = self.attention_layer.compute_weights(x, batch_index)
        return attention_scores
          
    def forward(self, x):
        var_states = []
        batch_index = []
        j = 0
        for i, s in enumerate(x):
            var_state = torch.tensor(s.reshape(-1, self.var_input_dim))
            var_states.extend(var_state)
            batch_index.extend(torch.repeat_interleave(torch.tensor(j), len(var_state)))  
            j += 1
        # concat all var_states
        var_states_var = torch.stack(var_states)
        #print("WEIGHTS", attention_weights.shape)

        if self.use_gpu:
            var_states_var = var_states_var.to(self.device, dtype=torch.float32)
            # change to float
            
        x = self.phi(var_states_var) 
        batch_index = torch.LongTensor(batch_index)
        if self.use_gpu:
            batch_index = batch_index.to(self.device)
    
        # Sum pooling layer
        #x_all = global_add_pool(x, batch_index) 
        #if self.use_cuda:
        #    x_all = x_all.cuda()

        # Apply repertoire-level attention
        x_all = self.attention_layer(x, batch_index)

        deep_set_out = self.rho(x_all)
        # output activation for classification
        return deep_set_out


class DeepSetClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, params=None):
        if params is None:
            # Set default parameters
            params = {
                'var_input_dim': 15,
                'batch_size': 2,
                'phi_hidden_dim': 128,
                'phi_output_dim': 128,
                'rho_hidden_dim': 128,
                'rho_output_dim': 128,
                'attention_dim': 64,
                'epochs': 1000,
                'learning_rate': 0.001,
                'dropout' : None

            }

        self.var_input_dim = params['var_input_dim']
        self.batch_size = params['batch_size']
        self.phi_hidden_dim = params['phi_hidden_dim']
        self.phi_output_dim = params['phi_output_dim']
        self.rho_hidden_dim = params['rho_hidden_dim']
        self.rho_output_dim = params['rho_output_dim']
        self.attention_dim = params['attention_dim']
        self.epochs = params['epochs']
        self.learning_rate = params['learning_rate']
        self.attention_type = params['attention_type']
        self.dropout = params['dropout'] if 'dropout' in params else None
        self.model = None
        

    def init(self):
        self.model = DeepSet(
            var_input_dim=self.var_input_dim,
            batch_size=self.batch_size,
            phi_hidden_dim=self.phi_hidden_dim,
            phi_output_dim=self.phi_output_dim,
            rho_hidden_dim=self.rho_hidden_dim,
            rho_output_dim=self.rho_output_dim,
            attention_dim=self.attention_dim, 
            attention_type=self.attention_type,
            dropout=self.dropout
        )
    def fit(self, X, y, use_gpu=False):
        # Create the DeepSet model
        self.model = DeepSet(
            var_input_dim=self.var_input_dim,
            batch_size=self.batch_size,
            phi_hidden_dim=self.phi_hidden_dim,
            phi_output_dim=self.phi_output_dim,
            rho_hidden_dim=self.rho_hidden_dim,
            rho_output_dim=self.rho_output_dim,
            attention_dim=self.attention_dim, 
            attention_type=self.attention_type,
            dropout=self.dropout
        )

      
        if platform == "darwin":
            use_gpu = torch.backends.mps.is_available()
            device = "mps" if torch.backends.mps.is_available() else "cpu"  
        else:
            use_gpu = torch.cuda.is_available()
            device = "cuda" if torch.cuda.is_available() else "cpu"

        if use_gpu:
            self.model = self.model.to(device, dtype=torch.float32)  # Convert model to float32

        # Define the optimizer and loss function
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.CrossEntropyLoss()

        X = np.array(X)
        print(X.shape)
        # Train the model
        for epoch in range(self.epochs):
            running_loss = 0.0
            num_batches = len(X) // self.batch_size
            for i in range(num_batches):
                # Randomly sample indices for the current batch
                indices = np.random.choice(range(len(X)), size=self.batch_size, replace=False)
                batch_inputs, batch_labels = [], []
                for idx in indices:
                    inputs, labels = X[idx], y[idx]
                    batch_inputs.append(inputs)
                    batch_labels.append(labels)

                # zero the parameter gradients
                optimizer.zero_grad()
                # forward + backward + optimize
                outputs = self.model(batch_inputs)
                loss = criterion(outputs, torch.tensor(batch_labels).to(device).long())
                loss.backward()
                optimizer.step()
                # print statistics
                running_loss += loss.item()
                # early stopping with loss < 0.0001
            if running_loss < 0.0005:
                print('Early stopping at epoch %d' % (epoch + 1))
                return self

            if epoch % 100 == 0:    # print every 100 mini-batches
                print('[%d, %5d] loss: %.4f' % (epoch + 1, i + 1, running_loss))
                running_loss = 0.0

        return self

    def predict(self, X):
        X = np.array(X)
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X)
            _, predicted = torch.max(outputs, 1)
            return predicted.cpu().numpy()
        
    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path))
        