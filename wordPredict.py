import os
import torch
import h5py
from math import sqrt, inf
from random import shuffle
from torch import nn, optim
from jsrs.exercise_codes.wordFreq import load


class Ngram(nn.Module):

    def __init__(self, vsize, context_size=3, wdim=32, hsize=128, bind_embed=True):
        super(Ngram, self).__init__()

        self.embed = nn.Embedding(vsize, wdim)
        self.ngram = context_size

        self.net = nn.Sequential(
            nn.Linear(wdim*context_size, hsize),
            nn.GELU(),
            nn.Linear(hsize, wdim, bias=False)
        )
        self.classifier = nn.Linear(wdim, vsize)

        if bind_embed:
            self.classifier.weight = self.embed.weight

    def forward(self, input):

        # input: (bsize, seql)
        # iemb: (bsize, seql, w_dim)
        iemb = self.embed(input)
        inar = []
        ndata = input.size(-1) - self.ngram

        # inar:(ngram, bsize, ndata, w_dim), like a tuple of ngram tensors
        for i in range(self.ngram):
            inar.append(iemb.narrow(1, i, ndata))

        inet = torch.cat(inar, dim=-1)

        # output: (bsize, ndata, vsize)
        output = self.classifier(self.net(inet))

        return output

    def decode(self, input, steps=50):

        # input: (bsize, context_size)
        _input = input
        rs = []
        bsize, nkeep, last_step = input.size(0), self.ngram - 1, steps - 1

        for i in range(steps):
            # output: (bsize, vsize)
            output = self.classifier(self.net(self.embed(_input).view(bsize, -1)))
            # wdind: (bsize)
            wdind = output.argmax(-1)
            rs.append(wdind)
            if i < last_step:
                _input = torch.cat((_input.narrow(1, 1, nkeep), wdind.unsqueeze(-1),),  dim=-1)
        # return: (bsize, steps)
        return torch.stack(rs, dim=-1)


class RNNLayer(nn.Module):

    def __init__(self, isize=32, osize=None, dropout=0.1):
        super(RNNLayer, self).__init__()

        _osize = osize if osize else isize

        self.net = nn.Sequential(
            nn.Linear(isize+_osize, _osize, bias=False),
            nn.LayerNorm(_osize),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        self.init_state = nn.Parameter(torch.zeros(1, _osize))

    def forward(self, input):

        bsize = input.size(0)
        _state = self.init_state.expand(bsize, -1)
        rs = []

        # input: (bsize, ndata, isize), inputu:(bsize, isize)
        for inputu in input.unbind(1):
            _state = self.net(torch.cat((inputu, _state), dim=-1))
            rs.append(_state)
        # rs:(ndata, bsize, isize), return:(bsize, ndata, isize)
        return torch.stack(rs, dim=1)

    def decode(self, input, state=None):
        # input:(bsize, isize)
        _state = self.init_state.expand(input.size(0), -1) if state is None else state
        out = self.net(torch.cat((input, _state), dim=-1))
        return out, out


class LSTMCell(nn.Module):

    def __init__(self, isize, osize=None, dropout=0.0):
        super(LSTMCell, self).__init__()

        _osize = isize if osize is None else osize

        self.trans = nn.Linear(isize+_osize, _osize*4, bias=False)
        self.normer = nn.LayerNorm([4, _osize])

        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout, inplace=True) if dropout > 0.0 else None

        self.osize = _osize

    def forward(self, inpute, state):

        _out, _cell = state

        osize = list(_out.size())
        osize.insert(-1,4)

        _comb = self.normer(self.trans(torch.cat((inpute, _out), dim=-1)).view(osize))
        (fg, og, ig,), hidden = _comb.narrow(-2, 0, 3).sigmoid().unbind(-2), self.act(_comb.select(-2, 3))

        if self.drop is not None:
            hidden = self.drop(hidden)

        _cell = fg * _cell + ig * hidden
        _out = og * _cell

        return _out, _cell


class LSTMLayer(nn.Module):

    def __init__(self, isize=32, osize=32, dropout=0.1):
        super(LSTMLayer, self).__init__()

        self.net = LSTMCell(isize=isize, osize=osize, dropout=dropout)

        self.init_state = nn.Parameter(torch.zeros(1, osize))
        self.init_cell = nn.Parameter(torch.zeros(1, osize))

    def forward(self, input):

        bsize = input.size(0)
        _state, _cell = self.init_state.expand(bsize, -1), self.init_cell.expand(bsize, -1)
        rs = []

        # input:(bsize, ndata, isize), inputu:(bsize, isize)
        for inputu in input.unbind(1):
            _state, _cell = self.net(inputu, (_state, _cell))
            rs.append(_state)

        return torch.stack(rs, dim=1)

    def decode(self, input, state=None):

        if state is None:
            bsize = input.size(0)
            _state, _cell = self.init_state.expand(bsize, -1), self.init_cell.expand(bsize, -1)
        else:
            _state, _cell = state
        _state, _cell = self.net(input, (_state, _cell))

        return _state, (_state, _cell)

    
class LSTMModel(nn.Module):
    
    def __init__(self, vsize, isize=32, osize=32, hsize=128, dropout=0.1, num_layer=2, bind_emb=True):
        super(LSTMModel, self).__init__()

        self.emb = nn.Embedding(vsize, isize)
        self.nets = nn.Sequential(*[LSTMLayer(isize = isize if i==0 else hsize, osize=hsize, dropout=dropout) for i in range(num_layer)])
        self.trans = nn.Linear(hsize, osize)
        self.classifier = nn.Linear(osize, vsize)

        if bind_emb:
            self.classifier.weight = self.emb.weight

    def forward(self, input):

        inpute = self.emb(input)
        return self.classifier(self.trans(self.nets(inpute)))

    def decode(self, input, steps=50):

        rs = []
        _state = {}
        for i in range(len(self.net)):
            _state[i] = None
        _input = input

        for i in range(steps):
            out = self.emb(_input)
            for j, net in enumerate(self.net):
                out, _state[j] = net.decode(out, state=_state[j])
            _input = out.argmax(-1)
            rs.append(_input)

        return torch.stack(rs, dim=-1)


class SelfAttn(nn.Module):

    def __init__(self, isize, hsize, osize, num_head=2, dropout=0.1):
        super(SelfAttn, self).__init__()

        self.attn_dim = hsize // num_head
        self.hsize = self.attn_dim * num_head
        self.num_head = num_head

        self.adaptor = nn.Linear(isize, self.hsize * 3, bias=False)

        self.outer = nn.Linear(self.hsize, osize, bias=False)

        self.normer = nn.Softmax(dim=-1)

        self.drop = nn.Dropout(dropout, inplace=False) if dropout > 0.0 else None

    def forward(self, iQ, mask=None):

        bsize, nquery = iQ.size()[:2]
        nheads = self.num_head
        adim = self.attn_dim

        real_iQ, real_iK, real_iV = self.adaptor(iQ).view(bsize, nquery, 3, nheads, adim).unbind(2)
        real_iQ, real_iK, real_iV = real_iQ.transpose(1, 2), real_iK.permute(0, 2, 3, 1), real_iV.transpose(1, 2)

        scores = real_iQ.matmul(real_iK)
        scores = scores / sqrt(adim)

        if mask is not None:
            scores.masked_fill_(mask.unsqueeze(1), -inf)

        scores = self.normer(scores)

        if self.drop is not None:
            scores = self.drop(scores)

        oMA = scores.matmul(real_iV).transpose(1, 2).contiguous()

        return self.outer(oMA.view(bsize, nquery, self.hsize))


class PositionwiseFF(nn.Module):

    def __init__(self, isize, hsize=None, dropout=0.0):
        super(PositionwiseFF, self).__init__()

        _hsize = isize * 4 if hsize is None else hsize

        self.net = nn.Sequential(
            nn.Linear(isize, _hsize),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(_hsize, isize, bias=False),
            nn.Dropout(dropout)
        ) if dropout > 0.0 else \
            nn.Sequential(
            nn.Linear(isize, _hsize),
            nn.GELU(),
            nn.Linear(_hsize, isize, bias=False)
        )

        self.normer = nn.LayerNorm(isize)

    def forward(self, x):

        _out = self.normer(x)

        out = self.net(_out)

        out = out + _out

        return out


class SA_FF_Layer(nn.Module):

    def __init__(self, isize, ahsize=None, fhsize=None, num_head=2, dropout=0.1, attn_dropout=0.1):
        super(SA_FF_Layer, self).__init__()

        _ahsize = isize if ahsize is None else ahsize
        _fhsize = _ahsize * 4 if fhsize is None else fhsize

        self.attn = SelfAttn(isize, _ahsize, isize, num_head=num_head, dropout=attn_dropout)
        self.ff = PositionwiseFF(isize, _fhsize, dropout=dropout)

        self.layer_normer = nn.LayerNorm(isize)

        self.drop = nn.Dropout(dropout) if dropout > 0.0 else None

    def forward(self, inputs, mask=None):

        _inputs = self.layer_normer(inputs)

        context = self.attn(_inputs, mask=mask)

        if self.drop is not None:
            context = self.drop(context)

        context = context + _inputs

        context = self.ff(context)

        return context


class SA_FF_Model(nn.Module):

    def __init__(self, vsize, isize=32, num_layer=2, num_head=2, xseql=256, fhsize=None, ahsize=None, dropout=0.1, attn_dropout=0.1, bind_emb=True):
        super(SA_FF_Model, self).__init__()

        _ahsize = isize if ahsize is None else ahsize
        _fhsize = _ahsize * 4 if fhsize is None else fhsize

        self.xseql = xseql
        self.register_buffer('mask', torch.ones(xseql, xseql, dtype=torch.bool).triu(1).unsqueeze(0))

        self.drop = nn.Dropout(dropout, inplace=True) if dropout > 0.0 else None
        self.emb = nn.Embedding(vsize, isize, padding_idx=0)
        self.nets = nn.ModuleList([SA_FF_Layer(isize, _ahsize, _fhsize, num_head, dropout, attn_dropout) for i in range(num_layer)])
        self.normer = nn.LayerNorm(isize)
        self.classifier = nn.Linear(isize, vsize)
        self.lsm = nn.LogSoftmax(-1)

        if bind_emb:
            self.classifier.weight = self.emb.weight

    def forward(self, inputs):
        # inpute:(bsize, seql), inputo:(bsize, nquery)
        inputs = inputs.narrow(1, 0, inputs.size(1)-1)
        nquery = inputs.size(-1)

        _inputs = self.emb(inputs)

        _inputs = _inputs * sqrt(_inputs.size(-1))

        if self.drop is not None:
            out = self.drop(_inputs)

        _mask = self._get_subsequent_mask(nquery)

        for net in self.nets:
            out = net(_inputs, mask=_mask)

        out = self.lsm(self.classifier(self.normer(out)))

        return out

    def _get_subsequent_mask(self, length):

        return self.mask.narrow(1, 0, length).narrow(2, 0, length) if length <= self.xseql else self.mask.new_ones(length, length).triu(1).unsqueeze(0)


def train(model, lossf, optimizer, batchs, vsize, wnum=3, epochs=50):

    train_list = list(batchs.keys())

    for epoch in range(epochs):

        shuffle(train_list)
        nbatch = 0

        for key in train_list:
            input = torch.LongTensor(batchs[key][()])
            target = input.narrow(1, wnum, input.size(-1) - wnum).reshape(-1)

            output = model(input).reshape(-1, vsize)

            loss = lossf(output, target)
            loss = loss/(target.ne(0).int().sum().item())
            print(epoch, nbatch, loss)

            loss.backward()
            if nbatch % 20 == 0:
                optimizer.step()
                optimizer.zero_grad()

            nbatch += 1
            if nbatch % 1000 == 0:
                torch.save(model.state_dict(), "D:/corpus/models/{}epochs{}batchs_model.en".format(epoch, nbatch))

        torch.save(model.state_dict(), "D:/corpus/models/{}epochs_final_model.en".format(epoch))


def predict(model, testf, r_vocab, seql=53, wnum=3):

    for batch in testf:
        input = torch.LongTensor(testf[batch][()])
        if input.size(-1) >= seql:
            break

    target = input.narrow(1, wnum, input.size(-1) - wnum)
    input = input.narrow(1, 0, wnum)

    with torch.no_grad():
        output = model.decode(input)

    for seq_o, seq_t in zip(output, target):
        print("output: " + " ".join([r_vocab[ind] for ind in seq_o.tolist() if ind in r_vocab]))
        print("target: " + " ".join([r_vocab[ind] for ind in seq_t.tolist() if ind in r_vocab]))


if __name__ == "__main__":

    dir_pre = "D:/corpus/pre/"
    dir_bpe = "D:/corpus/bpe/"
    dir_models = "D:/corpus/models/"
    batchs_p = os.path.join(dir_pre, "corpus_batchs.en")
    vocab_index_p = os.path.join(dir_pre, "vocab_index.en")
    statef = os.path.join(dir_models, "final_model.en")
    testfp = os.path.join(dir_pre, "newstest2014_batchs.en")

    f = h5py.File(batchs_p, "r")
    batchs = f["src"]
    nword = f["nword"][0]

    ft = h5py.File(testfp, "r")
    testf = ft["src"]

    vocab_index = load(vocab_index_p)
    r_vocab = {vocab_index[word]: word for word in vocab_index}

    # model = N_gram(nword)
    # model = LSTMModel(nword)
    model = SA_FF_Model(nword)

    for param in model.parameters():
        if param.dim() > 1:
            nn.init.xavier_uniform_(param)

    # model.load_state_dict(torch.load(statef))
    lossf = nn.CrossEntropyLoss(ignore_index=0, reduction="sum")
    optimizer = optim.Adam(model.parameters(), lr=1e-4, betas=(0.9,0.98,))

    train(model, lossf, optimizer, batchs, nword, 1)
    predict(model, testf, r_vocab, 51, 1)
