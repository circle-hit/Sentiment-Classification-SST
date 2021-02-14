import torch
from torch import dtype, long, nn
from star_Trans import *

class _Cls(nn.Module):
    def __init__(self, in_dim, num_cls, hid_dim, dropout=0.1):
        super(_Cls, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(hid_dim, num_cls),
        )

    def forward(self, x):
        h = self.fc(x)
        return h

class Cls_Task(nn.Module):
    r"""
    用于分类任务的Star-Transformer
    """

    def __init__(self, embedding_matrix, embedding_size, num_cls, max_len, voc_size,
                 hidden_size=300,
                 num_layers=4,
                 num_head=6,
                 head_dim=50,
                 pos_embedding_size=100,
                 cls_hidden_size=600,
                 emb_dropout=0.1,
                 dropout=0.1):
        r"""
        
        :param embed: 单词词典, 可以是 tuple, 包括(num_embedings, embedding_dim), 即
            embedding的大小和每个词的维度. 也可以传入 nn.Embedding 对象, 此时就以传入的对象作为embedding
        :param num_cls: 输出类别个数
        :param hidden_size: 模型中特征维度. Default: 300
        :param num_layers: 模型层数. Default: 4
        :param num_head: 模型中multi-head的head个数. Default: 8
        :param head_dim: 模型中multi-head中每个head特征维度. Default: 32
        :param max_len: 模型能接受的最大输入长度. Default: 512
        :param cls_hidden_size: 分类器隐层维度. Default: 600
        :param emb_dropout: 词嵌入的dropout概率. Default: 0.1
        :param dropout: 模型除词嵌入外的dropout概率. Default: 0.1
        """
        super(Cls_Task, self).__init__()
        self.enc = StarTransformer(embedding_matrix=embedding_matrix,
                                embedding_dim=embedding_size,
                                max_len=max_len,
                                vocab_size=voc_size,
                                hidden_size=hidden_size,
                                num_layers=num_layers,
                                num_head=num_head,
                                head_dim=head_dim,
                                pos_embedding_size=pos_embedding_size,
                                emb_dropout=emb_dropout,
                                dropout=dropout)
        self.cls = _Cls(hidden_size, num_cls, dropout=dropout)

    def forward(self, words, mask):
        r"""
        :param words: [batch, seq_len] 输入序列
        :param seq_len: [batch,] 输入序列的长度
        :return output: [batch, num_cls] 输出序列的分类的概率
        """
        nodes, relay = self.enc(words, mask)
        y = 0.5 * (relay + nodes.max(1)[0])
        output = nn.functional.softmax(self.cls(y),dim=1)  # [bsz, n_cls]
        return output

    def predict(self, words, seq_len):
        r"""
        :param words: [batch, seq_len] 输入序列
        :param seq_len: [batch,] 输入序列的长度
        :return output: [batch, num_cls] 输出序列的分类
        """
        y = self.forward(words, seq_len)
        _, pred = y.max(1)
        return pred

if __name__ == '__main__':
    # Testing Model
    torch.manual_seed(2)
    print("Testing star model...")
    x = torch.Tensor([[2,3,4,0,0,0,0,0,0,0,],[5,6,7,8,9,0,0,0,0,0]])
    x = x.long()
    seq_len = torch.Tensor([3,5])
    seq_len = seq_len.long()
    sentence = ["This is just a test to check whether the model work"]
    model1 = Cls_Task(None,300,5,5,20)
    output = model1(x,seq_len,10)
    print(output.shape)
    print("Test passed!")