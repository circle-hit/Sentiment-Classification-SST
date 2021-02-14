import torch
from torch import equal
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
from torch.utils import data
import torch.utils.data
import numpy as np

class StarTransformer(nn.Module):
    r"""
    带word embedding的Star-Transformer Encoder
    
    """

    def __init__(self, embedding_matrix,
                embedding_dim,
                max_len,
                vocab_size,
                hidden_size,
                num_layers,
                num_head,
                head_dim,
                pos_embedding_size,
                emb_dropout,
                dropout):
        r"""
        :param embedding_matrix: embedding矩阵
        :param embedding_dim: 词向量维度
        :param hidden_size: 模型中特征维度.
        :param num_layers: 模型层数.
        :param num_head: 模型中multi-head的head个数.
        :param head_dim: 模型中multi-head中每个head特征维度.
        :param max_len: 模型能接受的最大输入长度.
        :param emb_dropout: 词嵌入的dropout概率.
        :param dropout: 模型除词嵌入外的dropout概率.
        """
        super(StarTransformer, self).__init__()
        self.num_layers = num_layers
        self.norm = nn.ModuleList([nn.LayerNorm(hidden_size, eps=1e-6) for _ in range(self.num_layers)])
        # self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=True, padding_idx=0)
        if embedding_matrix is not None:
            self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=True, padding_idx=0)
        else:
            if vocab_size is None:
                raise Exception("Could not create new embedding: missing vocabulary size!")
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        self.emb_drop = nn.Dropout(emb_dropout)
        self.emb_fc = nn.Linear(embedding_dim, hidden_size)
        self.ring_att = nn.ModuleList(
            [_MSA1(hidden_size, nhead=num_head, head_dim=head_dim, dropout=0.0)
             for _ in range(self.num_layers)])
        self.star_att = nn.ModuleList(
            [_MSA2(hidden_size, nhead=num_head, head_dim=head_dim, dropout=0.0)
             for _ in range(self.num_layers)])
        
        self.pos_emb = nn.Embedding(max_len, embedding_dim)
        
    def forward(self, x, mask):
        r"""
        :param FloatTensor x: [batch, length] 输入的序列
        :param ByteTensor mask: [batch, length] 输入序列的padding mask, 在没有内容(padding 部分) 为 0,
            否则为 1
        :return: [batch, length, hidden] 编码后的输出序列
                [batch, hidden] 全局 relay 节点
        """
    
        def norm_func(f, x):
            # 用于进行dropout和normlization
            # B, H, L, 1
            return f(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        
        x = self.embedding(x)

        B, L, H = x.size()
        mask = (mask.eq(False))
        
        smask = torch.cat([torch.zeros(B, 1).byte().to(mask), mask], 1) # 在这里添加一个mask项的原因在于relay部分输入Tensor有拼接

        embed = x.view(B, H, L, 1)
        P = self.pos_emb(torch.arange(L, dtype=torch.long, device=embed.device).expand(1, L)).permute(0, 2, 1).contiguous()[:, :, :, None]  # 1 H L 1
        embed = embed + P 
        embed = norm_func(self.emb_drop, embed)

        #初始化nodes和relay
        nodes = embed                       #nodes:(B, H, L, 1)
        relay = embed.mean(2, keepdim=True) # relay:(B, H, 1, 1)
        
        ex_mask = mask[:, None, :, None].expand(B, H, L, 1)
        print(ex_mask.shape)
        r_embs = embed.view(B, H, 1, L)
        for i in range(self.num_layers):
            ax = torch.cat([r_embs, relay.expand(B, H, 1, L)], 2)
            nodes = F.leaky_relu(self.ring_att[i](norm_func(self.norm[i], nodes), ax=ax)) # nodes: (B, H, L, 1) 这里先进行norm再对nodes更新，与论文不同
            relay = F.leaky_relu(self.star_att[i](norm_func(self.norm[i],relay), torch.cat([relay, nodes], 2), smask)) # relay: (B, nhid, 1, 1)

            nodes = nodes.masked_fill_(ex_mask, 0)

        nodes = nodes.view(B, H, L).permute(0, 2, 1)
        relay = relay.view(B, H)
        return nodes, relay

class _MSA1(nn.Module):
    def __init__(self, nhid, nhead=6, head_dim=50, dropout=0.1):
        super(_MSA1, self).__init__()
        # Multi-head Self Attention Case 1, doing self-attention for small regions
        # To update satellite nodes
        # Due to the architecture of GPU, using hadamard production and summation are faster than dot production when unfold_size is very small
        self.WQ = nn.Conv2d(nhid, nhead * head_dim, 1)
        self.WK = nn.Conv2d(nhid, nhead * head_dim, 1)
        self.WV = nn.Conv2d(nhid, nhead * head_dim, 1)
        self.WO = nn.Conv2d(nhead * head_dim, nhid, 1)

        self.drop = nn.Dropout(dropout)

        # print('NUM_HEAD', nhead, 'DIM_HEAD', head_dim)
        self.nhid, self.nhead, self.head_dim, self.unfold_size = nhid, nhead, head_dim, 3

    def forward(self, x, ax=None):
        # x: B, H, L, 1, ax : B, H, X, L append features
        nhid, nhead, head_dim, unfold_size = self.nhid, self.nhead, self.head_dim, self.unfold_size
        B, H, L, _ = x.shape

        q, k, v = self.WQ(x), self.WK(x), self.WV(x)  # x: (B,H,L,1)
        # q: B, nhead * head_num, L, 1
        # k: B, nhead * head_num, L, 1
        # v: B, nhead * head_num, L, 1
    
        if ax is not None:
            aL = ax.shape[2]
            ak = self.WK(ax).view(B, nhead, head_dim, aL, L)
            av = self.WV(ax).view(B, nhead, head_dim, aL, L)
        
        q = q.view(B, nhead, head_dim, 1, L)
        k = F.unfold(k.view(B, nhead * head_dim, L, 1), (unfold_size, 1), padding=(unfold_size // 2, 0)) \
            .view(B, nhead, head_dim, unfold_size, L)
        v = F.unfold(v.view(B, nhead * head_dim, L, 1), (unfold_size, 1), padding=(unfold_size // 2, 0)) \
            .view(B, nhead, head_dim, unfold_size, L)
        
        if ax is not None:
            k = torch.cat([k, ak], 3)
            v = torch.cat([v, av], 3)

        alphas = self.drop(F.softmax((q * k).sum(2, keepdim=True) / np.sqrt(head_dim), 3))
        # alphas: (B, nhead, 1, unfold_size, L)
        att = (alphas * v).sum(3).view(B, nhead * head_dim, L, 1)
        
        ret = self.WO(att)
        # ret: (B, nhid, L, 1)
        return ret

class _MSA2(nn.Module):
    def __init__(self, nhid, nhead=6, head_dim=50, dropout=0.1):
        # Multi-head Self Attention Case 2, a broadcastable query for a sequence key and value
        # To update relay nodes
        super(_MSA2, self).__init__()
        self.WQ = nn.Conv2d(nhid, nhead * head_dim, 1)
        self.WK = nn.Conv2d(nhid, nhead * head_dim, 1)
        self.WV = nn.Conv2d(nhid, nhead * head_dim, 1)
        self.WO = nn.Conv2d(nhead * head_dim, nhid, 1)

        self.drop = nn.Dropout(dropout)

        # print('NUM_HEAD', nhead, 'DIM_HEAD', head_dim)
        self.nhid, self.nhead, self.head_dim, self.unfold_size = nhid, nhead, head_dim, 3

    def forward(self, x, y, mask=None):
        # x: B, H, 1, 1  y: B H L 1 mask: 
        nhid, nhead, head_dim, unfold_size = self.nhid, self.nhead, self.head_dim, self.unfold_size
        B, H, L, _ = y.shape

        q, k, v = self.WQ(x), self.WK(y), self.WV(y)

        q = q.view(B, nhead, 1, head_dim)  # B, H, 1, 1 -> B, N, 1, h
        k = k.view(B, nhead, head_dim, L)  # B, H, L, 1 -> B, N, h, L
        v = v.view(B, nhead, head_dim, L).permute(0, 1, 3, 2)  # B, H, L, 1 -> B, N, L, h
        pre_a = torch.matmul(q, k) / np.sqrt(head_dim) # pre_a: (B, nhead, 1, L)
        
        if mask is not None:
            mask.to(torch.device('cuda'))
            pre_a = pre_a.masked_fill(mask[:, None, None, :], -float('inf'))

        alphas = self.drop(F.softmax(pre_a, 3))  # B, N, 1, L
        att = torch.matmul(alphas, v).view(B, -1, 1, 1)  # B, N, 1, h -> B, N*h, 1, 1

        ret = self.WO(att)
        # ret: (B, nhid, 1, 1)
        return ret

def seq_len_to_mask(seq_len, max_len=None):
    r"""
    将一个表示sequence length的一维数组转换为二维的mask，不包含的位置为0。
    转变 1-d seq_len到2-d mask.
    .. code-block::
    
        >>> seq_len = torch.arange(2, 16)
        >>> mask = seq_len_to_mask(seq_len)
        >>> print(mask.size())
        torch.Size([14, 15])
        >>> seq_len = np.arange(2, 16)
        >>> mask = seq_len_to_mask(seq_len)
        >>> print(mask.shape)
        (14, 15)
        >>> seq_len = torch.arange(2, 16)
        >>> mask = seq_len_to_mask(seq_len, max_len=100)
        >>>print(mask.size())
        torch.Size([14, 100])
    :param np.ndarray,torch.LongTensor seq_len: shape将是(B,)
    :param int max_len: 将长度pad到这个长度。默认(None)使用的是seq_len中最长的长度。但在nn.DataParallel的场景下可能不同卡的seq_len会有
        区别，所以需要传入一个max_len使得mask的长度是pad到该长度。
    :return: np.ndarray, torch.Tensor 。shape将是(B, max_length)， 元素类似为bool或torch.uint8
    """
    if isinstance(seq_len, np.ndarray):
        assert len(np.shape(seq_len)) == 1, f"seq_len can only have one dimension, got {len(np.shape(seq_len))}."
        max_len = int(max_len) if max_len else int(seq_len.max())
        broad_cast_seq_len = np.tile(np.arange(max_len), (len(seq_len), 1))
        mask = broad_cast_seq_len < seq_len.reshape(-1, 1)

    elif isinstance(seq_len, torch.Tensor):
        assert seq_len.dim() == 1, f"seq_len can only have one dimension, got {seq_len.dim() == 1}."
        batch_size = seq_len.size(0)
        max_len = int(max_len) if max_len else seq_len.max().long()
        broad_cast_seq_len = torch.arange(max_len).expand(batch_size, -1).to(seq_len)
        mask = broad_cast_seq_len.lt(seq_len.unsqueeze(1))
    else:
        raise TypeError("Only support 1-d numpy.ndarray or 1-d torch.Tensor.")

    return mask

if __name__ == '__main__':
    # Testing Model
    torch.manual_seed(2)
    print("Testing star model...")
    x = torch.ones(10, 10).long()
    seq_len = torch.arange(1, 11)
    mask = seq_len_to_mask(seq_len)
    print(mask.shape)
    sentence = ["This is just a test to check whether the model work"]
    model1 = StarTransformer(None,10,10,1,1,10,10,0.5,0.1,10)
    output = model1(x,mask)
    print(output[0].shape, output[1].shape)
    print("Test passed!")