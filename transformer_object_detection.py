import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import math

from torchvision.models import resnet50

class PositionalEncoding(nn.Module):
    def __init__(self, emb_size: int, dropout: float, maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()

        den = torch.exp(-1 * torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)

        pos_embedding           = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2]  = torch.sin(pos * den)
        pos_embedding[:, 1::2]  = torch.cos(pos * den)
        pos_embedding           = pos_embedding.unsqueeze(-2)

        self.dropout        = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, tokens: Tensor) -> Tensor:
        return self.dropout(tokens + self.pos_embedding[:tokens.size(0), :])

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size: int):
        super(TokenEmbedding, self).__init__()

        self.embedding  = nn.Embedding(vocab_size, emb_size)

    def forward(self, tokens: Tensor) -> Tensor:
        return self.embedding(tokens.long())

class MultiHeadAttention(nn.Module):    
    def __init__(self, heads: int, d_model: int):        
        super(MultiHeadAttention, self).__init__()

        assert d_model % heads == 0
        self.d_k    = d_model // heads
        self.heads  = heads

        self.dropout    = nn.Dropout(0.1)
        self.query      = nn.Linear(d_model, d_model)
        self.key        = nn.Linear(d_model, d_model)
        self.value      = nn.Linear(d_model, d_model)
        self.out        = nn.Linear(d_model, d_model)
        
    def forward(self, query: Tensor, key: Tensor, value: Tensor, mask: Tensor = None) -> Tensor:
        query   = self.query(query)
        key     = self.key(key)        
        value   = self.value(value)   
        
        query   = query.view(query.shape[0], self.heads, -1, self.d_k)
        key     = key.view(key.shape[0], self.heads, -1, self.d_k)
        value   = value.view(value.shape[0], self.heads, -1, self.d_k)
       
        scores      = torch.matmul(query, key.transpose(2, 3))
        scores      = scores / math.sqrt(query.size(-1))
        
        if mask is not None:
            min_type_value  = torch.finfo(scores.dtype).min
            scores  = scores.masked_fill(mask == 0, min_type_value)
             
        weights     = F.softmax(scores, dim = -1)
        weights     = self.dropout(weights)

        context     = torch.matmul(weights, value)
        context     = context.transpose(1, 2).flatten(2)

        interacted  = self.out(context)
        return interacted

class FeedForward(nn.Module):
    def __init__(self, d_model: int, middle_dim: int = 2048):
        super(FeedForward, self).__init__()
        
        self.fc1        = nn.Linear(d_model, middle_dim)
        self.fc2        = nn.Linear(middle_dim, d_model)
        self.dropout    = nn.Dropout(0.1)

    def forward(self, x: Tensor) -> Tensor:
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, heads: int):
        super(EncoderLayer, self).__init__()

        self.layernorm      = nn.LayerNorm(d_model)
        self.self_multihead = MultiHeadAttention(heads, d_model)
        self.feed_forward   = FeedForward(d_model)
        self.dropout        = nn.Dropout(0.1)

    def forward(self, embeddings: Tensor, mask: Tensor) -> Tensor:
        interacted          = self.dropout(self.self_multihead(embeddings, embeddings, embeddings, mask))
        interacted          = self.layernorm(interacted + embeddings)
        feed_forward_out    = self.dropout(self.feed_forward(interacted))
        encoded             = self.layernorm(feed_forward_out + interacted)
        return encoded

class DecoderLayer(nn.Module):    
    def __init__(self, d_model: int, heads: int):
        super(DecoderLayer, self).__init__()

        self.layernorm      = nn.LayerNorm(d_model)
        self.self_multihead = MultiHeadAttention(heads, d_model)
        self.src_multihead  = MultiHeadAttention(heads, d_model)
        self.feed_forward   = FeedForward(d_model)
        self.dropout        = nn.Dropout(0.1)
        
    def forward(self, embeddings: Tensor, encoded: Tensor, target_mask: Tensor) -> Tensor:
        query               = self.dropout(self.self_multihead(embeddings, embeddings, embeddings, target_mask))
        query               = self.layernorm(query + embeddings)
        interacted          = self.dropout(self.src_multihead(query, encoded, encoded, None))
        interacted          = self.layernorm(interacted + query)
        feed_forward_out    = self.dropout(self.feed_forward(interacted))
        decoded             = self.layernorm(feed_forward_out + interacted)
        return decoded

class Transformer(nn.Module):    
    def __init__(self, d_model: int, heads: int, num_layers: int, vocab_size: int, dropout: float = 0.1):
        super(Transformer, self).__init__()
        
        self.d_model        = d_model
        self.vocab_size     = vocab_size
        self.embedding      = TokenEmbedding(self.vocab_size, d_model)
        self.pos_encoding   = PositionalEncoding(d_model, dropout = dropout)
        self.encoder        = nn.ModuleList([EncoderLayer(d_model, heads) for _ in range(num_layers)])
        self.decoder        = nn.ModuleList([DecoderLayer(d_model, heads) for _ in range(num_layers)])
        
    def encode(self, src_words: Tensor, src_mask: Tensor) -> Tensor:
        for layer in self.encoder:
            src_words = layer(src_words, src_mask)
        return src_words
    
    def decode(self, target_words: Tensor, target_mask: Tensor, src_embeddings: Tensor) -> Tensor:
        for layer in self.decoder:
            target_words = layer(target_words, src_embeddings, target_mask)

        return target_words
        
    def forward(self, src_words: Tensor, target_words: Tensor, src_mask: Tensor, target_mask: Tensor) -> Tensor:
        src_words   = self.pos_encoding(self.embedding(src_words))

        encoded     = self.encode(src_words, src_mask)
        decoded     = self.decode(target_words, target_mask, encoded)
        return decoded

class DETR(nn.Module):
    def __init__(self, num_classes: int, hidden_dim: int, nheads: int, num_encoder_layers: int, num_decoder_layers: int):
        super().__init__()
        # We take only convolutional layers from ResNet-50 model
        self.backbone       = nn.Sequential(*list(resnet50(pretrained=True).children())[:-2])
        self.conv           = nn.Conv2d(2048, hidden_dim, 1)
        self.transformer    = nn.Transformer(hidden_dim, nheads, num_encoder_layers, num_decoder_layers)
        self.linear_class   = nn.Linear(hidden_dim, num_classes + 1)
        self.linear_bbox    = nn.Linear(hidden_dim, 4)
        self.query_pos      = nn.Parameter(torch.rand(100, hidden_dim))
        self.row_embed      = nn.Parameter(torch.rand(50, hidden_dim // 2))
        self.col_embed      = nn.Parameter(torch.rand(50, hidden_dim // 2))

    def forward(self, inputs: Tensor) -> Tensor:
        x       = self.backbone(inputs)
        h       = self.conv(x)
        H, W    = h.shape[-2:]
        pos     = torch.cat([
            self.col_embed[:W].unsqueeze(0).repeat(H, 1, 1),
            self.row_embed[:H].unsqueeze(1).repeat(1, W, 1),
        ], dim=-1).flatten(0, 1).unsqueeze(1)
        h       = self.transformer(pos + h.flatten(2).permute(2, 0, 1), self.query_pos.unsqueeze(1))
        return self.linear_class(h), self.linear_bbox(h).sigmoid()