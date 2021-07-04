import os
import numpy
import h5py
import torch
from random import shuffle


hdf5_data_compression = "gzip"
hdf5_data_compression_level = 9
h5datawargs = {} if hdf5_data_compression is None else {"compression": hdf5_data_compression, "compression_opts": hdf5_data_compression_level, "shuffle":True}

use_unk = False

pad_id, sos_id, eos_id, unk_id = 0, 1, 2, 4

def clean_list(lin):

    return [tmpu for tmpu in lin if tmpu]

def clean_liststr_lentok(lin):

    rs = [tmpu for tmpu in lin if tmpu]

    return " ".join(rs), len(rs)

def iter_dict_sort(dict_in, reverse=False):

    d_keys = list(dict_in.keys())
    d_keys.sort(reverse=reverse)
    for d_key in d_keys:
        d_v = dict_in[d_key]
        if isinstance(d_v, dict):
            for _item in iter_dict_sort(d_v):
                yield _item
        else:
            if len(d_v) > 1:
                shuffle(list(d_v))
            yield d_v

def clean_list_iter(lin):

    for lu in lin:
        if lu:
            yield lu

def list_reader(fname):

    with open(fname, "rb") as frd:
        for line in frd:
            tmp = line.strip()
            if tmp:
                tmp = clean_list(tmp.decode("utf-8").split())
                yield tmp

def get_bsize(maxlen, maxtoken, maxbsize):

    rs = max(maxtoken // maxlen, 1)
    if (rs % 2 == 1) and (rs > 1):
        rs -= 1

    return min(rs, maxbsize)

def no_unk_mapper(vcb, ltm, prompt=True):

    if prompt:
        rs = []
        for wd in ltm:
            if wd in vcb:
                rs.append(vcb[wd])

        return rs
    else:
        return [vcb[wd] for wd in ltm if wd in vcb]

def map_batch(i_d, vocabi):

    global use_unk, unk_id

    if isinstance(i_d[0], (tuple, list,)):
        return [map_batch(idu, vocabi)[0] for idu in i_d], 0
    else:
        rsi = []
        rsi.extend([vocabi.get(wd, unk_id) for wd in i_d] if use_unk else no_unk_mapper(vocabi, i_d))#[vocabi[wd] for wd in i_d if wd in vocabi]       
        return rsi, 0

def pad_batch(i_d, mlen_i):

    global pad_id
    if isinstance(i_d[0], (tuple, list,)):
        return [pad_batch(idu, mlen_i) for idu in i_d]
    else:
        curlen = len(i_d)
        if curlen < mlen_i:
            i_d.extend([pad_id for i in range(mlen_i - curlen)])
    return i_d

def ldvocab(vfile):

    rs = {"<pad>":0}
    cwd = 1
    for data in list_reader(vfile):
        wd = data[0]
        rs[wd] = cwd
        cwd += 1
    return rs, cwd

def create_train_test(infp, trainfp, testfp):

    ndata = 0
    ens = "\n".encode("utf-8")

    with open(infp, 'rb') as inf, open(trainfp, 'wb') as trainf, open(testfp, 'wb') as testf:
        for line in inf:
            if line:
                tmp = clean_list(line.decode("utf-8").strip().split())
                if len(tmp) <= 128:
                    ndata += 1
                    if ndata <= 4096:
                        trainf.write(" ".join(tmp).encode("utf-8"))
                        trainf.write(ens)
                    elif ndata <= 8192:
                        testf.write(" ".join(tmp).encode("utf-8"))
                        testf.write(ens)
                    else:
                        break
                else:
                    continue

def batch_loader(finput, max_tokens, max_len, min_len, min_bsize):

    rsi = []
    nd = mlen_i = b_tokens = 0

    for i_d in list_reader(finput):
        lgth = len(i_d)

        if (nd < min_bsize) or (lgth <= max_len and lgth >= min_len and b_tokens <= max_tokens):
            rsi.append(i_d)
            if lgth > mlen_i:
                mlen_i = lgth
            nd += 1
            b_tokens += lgth
        else:
            yield rsi, mlen_i
            rsi = [i_d]
            mlen_i = lgth
            nd = 1
            b_tokens = 0
    if rsi:
        yield rsi, mlen_i

def batch_mapper(finput, vocabi, max_tokens, max_len, min_len, min_bsize):

    for i_d, mlen_i in batch_loader(finput, max_tokens, max_len, min_len, min_bsize):
        rsi, extok_i = map_batch(i_d, vocabi)
        yield rsi, mlen_i + extok_i

def batch_padder(finput, vocabi, max_tokens, max_len, min_len, min_bsize):
    for i_d, mlen_i in batch_mapper(finput, vocabi, max_tokens, max_len, min_len, min_bsize):
        yield pad_batch(i_d, mlen_i)

def batch_creator(finput, vocabi, max_tokens, max_len, min_len, min_bsize):

    rsi = []
    nd = mlen_i = b_tokens = 0

    for i_d in list_reader(finput):
        lgth = len(i_d)

        if (nd < min_bsize) or (lgth <= max_len and lgth >= min_len and (b_tokens + lgth) <= max_tokens):
            rsi.append(i_d)
            if lgth > mlen_i:
                mlen_i = lgth
            nd += 1
            b_tokens += lgth
        else:
            batch = []
            for seq in rsi:
                ls = []
                for word in seq:
                    if word in vocabi:
                        ls.append(vocabi[word])
                batch.append(ls)
            for seq in batch:
                curlen = len(seq)
                if curlen < mlen_i:
                    seq.extend([0 for i in range(mlen_i - curlen)])
            yield batch
            rsi = [i_d]
            mlen_i = lgth
            nd = 1
            b_tokens = lgth
    if rsi:
        batch = []
        ls = []
        for seq in rsi:
            for word in seq:
                if word in vocabi:
                    ls.append(vocabi[word])
            batch.append(ls)
            ls = []
        for seq in batch:
            curlen = len(seq)
            if curlen < mlen_i:
                seq.extend([0 for i in range(mlen_i - curlen)])
        yield batch

def sortdata(srcfs, tgtfs, min_len=4, max_len=256):

    data = {}

    with open(srcfs, "rb") as fs:
        for ls in fs:
            ls = ls.strip()
            if ls:
                ls, lgth = clean_liststr_lentok(ls.decode("utf-8").split())
                if lgth <= max_len and lgth >= min_len:
                    if lgth in data:
                        if ls not in data[lgth]:
                            data[lgth].add(ls)
                    else:
                        data[lgth] = {ls}

    ens = "\n".encode("utf-8")

    with open(tgtfs, "wb") as fs:
        for tmp in iter_dict_sort(data):
            fs.write("\n".join(tmp).encode("utf-8"))
            fs.write(ens)

def mapvocab(srcf, rsf):

    vocab = {}

    with open(srcf, "rb") as f:
        for line in f:
            tmp = line.strip()
            if tmp:
                for token in clean_list_iter(tmp.decode("utf-8").split()):
                    vocab[token] = vocab.get(token, 0) + 1

    r_vocab = {}
    for k, v in vocab.items():
        if v not in r_vocab:
            r_vocab[v]=[str(v), k]
        else:
            r_vocab[v].append(k)

    freqs = list(r_vocab.keys())
    freqs.sort(reverse=True)

    ens = "\n".encode("utf-8")
    with open(rsf, "wb") as f:
        for freq in freqs:
            cdata = r_vocab[freq]
            f.write(" ".join(cdata).encode("utf-8"))
            f.write(ens)

def transbatchs(finput, fvocab_i, frs, newvocab, max_tokens = 2560, max_len = 256, min_len = 4, min_bsize = 1):
    vcbi, nwordi = ldvocab(fvocab_i)
    #with open(newvocab, "wb") as fwt:
    #   fwt.write(repr(vcbi).encode("utf-8"))

    rsf = h5py.File(frs,'w')
    src_grp = rsf.create_group("src")
    curd = 0
    
    for i_d in batch_creator(finput, vcbi, max_tokens, max_len, min_len, min_bsize):
        rid = numpy.array(i_d, dtype = numpy.int32)
        #rld = numpy.array(ld, dtype = numpy.int32)
        wid = str(curd)
        print(curd)
        print(rid)
        src_grp.create_dataset(wid, data=rid, **h5datawargs)
        #rsf["l" + wid] = rld
        curd += 1

    rsf["ndata"] = numpy.array([curd], dtype=numpy.int32)
    rsf["nword"] = numpy.array([nwordi], dtype=numpy.int32)

    rsf.close()
    print("Number of batches: %d\nSource Vocabulary Size: %d" % (curd, nwordi))

from math import inf

if __name__ == "__main__":
    '''
    dir_bpe = "D:/corpus/bpe/"
    dir_pre = "D:/corpus/pre/"
    finput = os.path.join(dir_bpe, "newstest2014_bpe.en")
    tgtfs = os.path.join(dir_pre, "newstest2014_sorted.en")
    frs = os.path.join(dir_pre, "newstest2014_batchs.en")
    vocab = os.path.join(dir_bpe, "vocab_bpe.en")
    vocab_index = os.path.join(dir_pre, "vocab_index.en")

    sortdata(finput, tgtfs)
    transbatchs(tgtfs, vocab, frs, vocab_index)
    '''
    '''
    f = h5py.File(frs, 'r')
    src = f["src"]
    print(src['70913'])
    
    x = torch.randn(5,3)
    x = [[1,2,3],[2,3,4]]
    x = torch.tensor(x)
    y = torch.cat((x,), dim=-1)
    print(x)
    print(y)
    '''
    '''
    x = torch.tensor([[1,2,3], [3,4,5]])
    print(x)
    print(torch.cat((x, x),dim=-1))
    '''

    inf = inf
    print(-inf)
    ls = [1, 2, 3]
    ls.insert(-1, 4)
    print(ls)
