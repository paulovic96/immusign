import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self,
                 dim,
                 hidden_dim,
                 dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class ExtractionModule(nn.Module):
    def __init__(self,
                 dim,
                 num_heads=8):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.kv = nn.Linear(dim, dim * 2, bias=False)
        self.q = nn.Linear(dim, dim, bias=False)
        self.proj = nn.Linear(dim, dim)

    def forward(self, target_token, clone_tokens, return_attention=False):
        B, N, C = clone_tokens.shape
        q = self.q(target_token).reshape(B, 1, 1, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        kv = self.kv(clone_tokens).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        attn = attn.softmax(dim=-1)
        result = (attn @ v).transpose(1, 2).reshape(B, 1, C)
        result = self.proj(result)
        if return_attention == True:
             return target_token + result, attn
        return target_token + result


class InjectionModule(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.mlp = MLP(2 * dim, 4 * dim)
        self.proj = nn.Linear(2 * dim, dim)

    def forward(self, target_token, clone_tokens):
        B, N, C = clone_tokens.shape
        target_token = target_token.expand(B, N, -1)
        result = torch.cat((target_token, clone_tokens), dim=-1)
        result = self.mlp(result)
        result = self.proj(result)
        return clone_tokens + result


class SimpleInteractionModule(nn.Module):
    def __init__(self,
                 dim,
                 num_heads=8):
        super().__init__()
        self.extraction = ExtractionModule(dim, num_heads)
        self.mlp = MLP(dim, 4 * dim)

    def forward(self, target_token, clone_tokens):
        target_token = self.extraction(target_token, clone_tokens)
        clone_tokens = self.mlp(clone_tokens)
        return target_token, clone_tokens


class SophisticatedInteractionModule(nn.Module):
    def __init__(self,
                 dim,
                 num_heads=8):
        super().__init__()
        self.extraction = ExtractionModule(dim, num_heads)
        self.injection = InjectionModule(dim)

    def forward(self, target_token, clone_tokens):
        target_token = self.extraction(target_token, clone_tokens)
        clone_tokens = self.injection(target_token, clone_tokens)
        return target_token, clone_tokens


class TransformerBlock(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 interaction='simple'
                 ):
        super().__init__()
        if interaction == 'simple':
            self.interaction = SimpleInteractionModule(dim, num_heads)
        else:
            self.interaction = SophisticatedInteractionModule(dim, num_heads)
        self.mlp = MLP(dim, int(dim * mlp_ratio))

    def forward(self, target_token, clone_tokens):
        target_token, clone_tokens = self.interaction(target_token, clone_tokens)
        clone_tokens = clone_tokens + self.mlp(clone_tokens)
        return target_token, clone_tokens
