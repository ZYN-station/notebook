import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy
import math
import numpy as np
from torch.autograd import Variable

class Embedding(nn.Module):
    def __init__(self,d_model,vocab):
        super(Embedding,self).__init__()
        self.d_model = d_model
        self.vocab = vocab
        self.lut = nn.Embedding(d_model,vocab)
    def forward(self,x):
        return self.lut(x) * math.sqrt(self.d_model)

class PositionEncoding(nn.Module):
    def __init__(self,d_model,dropout = 0.1,max_len = 60):
        super(PositionEncoding,self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0,max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0,d_model,2) *
                             -math.log(10000)/d_model)
        pe[:,0::2] = torch.sin(position*div_term)
        pe[:,1::2] = torch.cos(position*div_term)
        self.dropout = nn.Dropout(p=dropout)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe',pe)
    def forward(self,x):
        return self.dropout(x+Variable(self.pe[:,x.size(1)],requires_grad = False))

"""掩码张量"""
def subsequent_mask(size):
    attn_shape = (1,size,size)
    subsequent_mask = np.triu(np.ones(attn_shape),k = 1).astype('unit8')
    return torch.from_numpy(1-subsequent_mask)

"""注意力机制"""
def attention(query,key,value,mask=None,dropout = None):
    # query,key,value,三个三维张量
    d_k = query.size(-1)
    scores = torch.matmul(query,key.transpose(-2,-1))/math.sqrt(d_k)
    if mask is not None:
        scores = torch.masked_fill(mask==0,-1e-9)
    p_attn = F.softmax(scores,dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn,value),p_attn
"""克隆"""
import copy
def clones(model,N):
    return nn.ModuleList([copy.deepcopy(model)for _ in range(N)])

#规范化层
class LayerNorm(nn.Module):
    def __init__(self,features,eps=1e-6):
        super(LayerNorm,self).__init__()
        self.a = torch.ones(features)
        self.b = torch.zeros(features)
        self.eps = eps
    def forward(self,x):
        mean = x.mean(-1,keepdim=True)
        std = x.std(-1,keepdim = True)
        return (x-mean)*self.a/(std+self.eps)+self.b


class PositionwiseFeedForward(nn.Module):
    def __init__(self,d_model,d_ff,dropout = 0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.w1 = nn.Linear(d_model,d_ff)
        self.w2 = nn.Linear(d_ff,d_model)
        self.dropout = nn.Dropout(p = dropout)
    def forward(self,x):
        return self.w2(self.dropout(F.relu(self.w1(x))))

def attention(query, key, value, mask=None, dropout=None):
    #词嵌入维度
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e-9)
    p_attn = F.softmax(scores, dim=-1)

    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

class MultiHeadedAttention(nn.Module):
    def __init__(self,head,embedding_dim,dropout = 0.1):
        super(MultiHeadedAttention,self).__init__()
        assert embedding_dim % head == 0
        self.d_k = embedding_dim // head
        self.head = head
        self.linears = clones(nn.Linear(embedding_dim,embedding_dim), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        batch_size = query.size(0)
        query, key, value = [model(x).view(batch_size,-1,self.head,self.d_k)
                             for model,x in zip(self.linears,[query, key, value])]
        x,self.attn = attention(query,key,value,mask=mask,dropout=self.dropout)
        x = x.transpose(1,2).contiguous().view(batch_size,-1,self.head*self.d_k)
        return self.linears[-1](x)

class SublayerConnection(nn.Module):
    def __init__(self, size, dropout = 0.1):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(p=dropout)
    def forward(self, x, sublayer):
        return x+self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    # size:词嵌入维度大小
    # self_attn：多头自注意力子层实例化对象
    # feed_forward: 前馈全连接层实例化对象
    def __init__(self, size, self_attn, feed_forward, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward

        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        x = self.sublayer[1](x, self.feed_forward)

class Encoder(nn.Module):
    def __init__(self,layer,N):
        super(Encoder,self).__init__()
        self.layers = clones(layer,N)
        self.norm = LayerNorm(layer.size)
    def forward(self,x,mask=None):
        for layer in self.layers:
            x = layer(x,mask)
        return self.norm(x)

class DecoderLayer(nn.Module):
    """
    size:词维度大小
    src_attn:多头注意力机制
    self_attn:多头自注意力机制
    feed_forward:前馈连接层
    """
    def __init__(self,size,src_attn,self_attn,feed_forward,dropout = 0.1):
        super(DecoderLayer, self).__init__()
        # 在初始化函数中， 主要就是将这些输入传到类中
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        # 按照结构图使用clones函数克隆三个子层连接对象.
        self.sublayer = clones(SublayerConnection(size, dropout), 3)
    def forward(self,x,memory,source_mask,target_mask):
        #编码器的输出
        m = memory
        x = self.sublayers[0](x,lambda x:self.self_attn(x,x,x,source_mask))
        x = self.sublayers[1](x,lambda x:self.self_attn(x,m,m,target_mask))
        return self.sublayers[2](x,self.feed_forward)

class Decoder(nn.Module):
    def __init__(self,layer,N):
        super(Decoder,self).__init__()
        self.layers = clones(layer,N)
        self.norm = LayerNorm(layer.size)
    def forward(self,x,memory,source_mask,target_mask):
        for layer in self.layers:
            x = layer(x,memory,source_mask,target_mask)
        return self.norm(x)

class Generator(nn.Module):
    def __init__(self, d_model, vocab_size):
        super(Generator, self).__init__()
        self.project = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        return F.log_softmax(self.project(x), dim=-1)

class EncoderDecoder(nn.Module):
    def __init__(self,encoder,decoder,source_embed,target_embed,generator):
        super(EncoderDecoder,self).__init__()
        #source_embed源数据嵌入函数, target_embed目标数据嵌入函数
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = source_embed
        self.tgt_embed = target_embed
        self.generator = generator
    def forward(self,source,target,source_mask,target_mask):
        return self.decode(self.encode(source,source_mask),source_mask,target,target_mask)
    def encode(self,source,source_mask):
        return self.encoder(self.src_embed(source),source_mask)
    def decode(self,memory,source_mask,target,target_mask):
        return self.decoder(memory,source_mask,self.tgt_embed(target),target_mask)


def make_mode(source_vocab, target_vocab, N=6, d_model=512, d_ff=2048, head=8, dropout=0.1):
    """该函数用来构建模型, 有7个参数，分别是源数据特征(词汇)总数，目标数据特征(词汇)总数，
       编码器和解码器堆叠数，词向量映射维度，前馈全连接网络中变换矩阵的维度，
       多头注意力结构中的多头数，以及置零比率dropout."""
    c = copy.deepcopy
    # 多头注意力机制
    attn = MultiHeadedAttention(head, d_model, dropout)

    # 位置编码
    position = PositionEncoding(d_model, dropout)

    # 前馈连接层
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)

    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        nn.Sequential(Embedding(d_model, source_vocab), c(position)),
        nn.Sequential(Embedding(d_model, target_vocab), c(position)),
        Generator(d_model, target_vocab)
    )

    # 模型结构完成后，接下来就是初始化模型中的参数，比如线性层中的变换矩阵
    # 这里一但判断参数的维度大于1，则会将其初始化成一个服从均匀分布的矩阵，
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)
    return model
