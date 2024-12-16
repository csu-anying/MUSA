# coding:utf-8
import os
import numpy
from collections import Counter
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import platform
import random
from model_train import PKL_PATH,TOSENTENCE,RESAMP,DATA_SET,IOBES,DEV_TRAIN,DIC
from dic2vec import dembedding_mat,dic_cnt
M = 5
N = 5
# DICC = DIC
DATA_PATH = './data/'+DATA_SET+'/'

if DATA_SET == 'conll':
    if IOBES == 'iob':
        chunk_tags_ = ['O', 'B-PER', 'I-PER', 'B-LOC', 'I-LOC', "B-ORG", "I-ORG"]
    else:
        chunk_tags_ = ['O', 'B-PER', 'I-PER','E-PER','S-PER', 'B-LOC', 'I-LOC','E-LOC','S-LOC', "B-ORG", "I-ORG",'E-ORG','S-ORG']
elif DATA_SET == 'ccks':
    if IOBES == 'iob':
        chunk_tags_ = ['O', 'B-BODY', 'I-BODY', 'B-SIGNS', 'I-SIGNS',
                       "B-CHECK", "I-CHECK", "B-DISEASE", "I-DISEASE",
                       "B-TREATMENT", "I-TREATMENT"]
    else:
        chunk_tags_ = ['O', 'B-BODY', 'I-BODY', 'E-BODY', 'S-BODY',
                       'B-SIGNS', 'I-SIGNS', 'E-SIGNS', 'S-SIGNS',
                       "B-CHECK", "I-CHECK", "E-CHECK", "S-CHECK",
                       "B-DISEASE", "I-DISEASE", "E-DISEASE",
                       "B-TREATMENT", "I-TREATMENT", "E-TREATMENT"]

elif DATA_SET == 'ccks2':
    if IOBES == 'iob':
        chunk_tags_ = ['O', 'B-AP', 'I-AP',
                       'B-IP', 'I-IP', 
                       "B-SD", "I-SD", 
                       "B-OP", "I-OP", 
                       "B-DG", "I-DG", ]
    else:
        chunk_tags_ = ['O', 'B-AP', 'I-AP', 'E-AP', 'S-AP',
                       'B-IP', 'I-IP', 'E-IP', 'S-IP',
                       "B-SD", "I-SD", "E-SD", "S-SD",
                       "B-OP", "I-OP", "E-OP", "S-OP",
                       "B-DG", "I-DG", "E-DG","S-DG"]

if IOBES == 'iobes':
    if DIC:
        # 加入词典特征
        TRAIN_PAH = DATA_PATH + 'train_dev_dic_set2.txt'
        TEST_PATH = DATA_PATH + 'test_dic_set2.txt'
    else:
        if DEV_TRAIN == 'dev_train':
            TRAIN_PAH = DATA_PATH+'train_dev_iobes_set.txt'
        else:
            TRAIN_PAH = DATA_PATH+'train_iobes_set.txt'
            DEV_PAH = DATA_PATH+'dev_iobes_set.txt'
        TEST_PATH = DATA_PATH+'test_iobes_set.txt'
        
elif IOBES == 'iob':
    if DEV_TRAIN == 'dev_train':
        TRAIN_PAH = DATA_PATH+'train_dev_set.txt'
    else:
        TRAIN_PAH = DATA_PATH+'train_set.txt'
        DEV_PAH = DATA_PATH+'dev_set.txt'
    TEST_PATH = DATA_PATH+'test_set.txt'


def load_data():
    #
    chunk_tags = chunk_tags_
    train = _parse_data(open(TRAIN_PAH, 'rb'))
    test = _parse_data(open(TEST_PATH, 'rb'))
    print('tosentence',TOSENTENCE)
    if TOSENTENCE:
        print('tosentence')
        train = parse_data_sentence(train)
        test = parse_data_sentence(test)

    if RESAMP == 'resamp':
        train = re_sampling(train,M,N)

    if DIC:
        # dic2vec
        if DATA_SET=='ccks':
            vocab, vocab0, vocab1 = dic_cnt('train_dic_set2.txt',DATA_SET)
        elif DATA_SET=='ccks2':
            vocab, vocab0, vocab1 = dic_cnt('train_dic_set2.txt',DATA_SET)
        with open(PKL_PATH, 'wb') as outp:
            pickle.dump((vocab,vocab0, vocab1, chunk_tags), outp)

        # 已经转换为数字序列
        train = _process_data(train, vocab1, chunk_tags)
        test = _process_data(test, vocab1, chunk_tags)
        return train, test, (vocab, vocab0, vocab1, chunk_tags)
    else:
        # 训练集中提取词表
        word_counts = Counter(row[0].lower() for sample in train for row in sample)
        # 获取词表 {词：词频>2}
        vocab = [w for w, f in iter(word_counts.items()) if f >= 5]

        # save initial config data
        with open(PKL_PATH, 'wb') as outp:
            pickle.dump((vocab, chunk_tags), outp)

        # 已经转换为数字序列
        train = _process_data(train, vocab, chunk_tags)
        test = _process_data(test, vocab, chunk_tags)
        return train, test, (vocab, chunk_tags)

def load_dev_data():
    # 单独加载dev文件
    # chunk_tags = ['O', 'B-PER', 'I-PER', 'B-LOC', 'I-LOC', "B-ORG", "I-ORG"]
    chunk_tags = chunk_tags_
    train = _parse_data(open(TRAIN_PAH, 'rb'))
    dev = _parse_data(open(DEV_PAH, 'rb'))
    test = _parse_data(open(TEST_PATH, 'rb'))
    if TOSENTENCE:
        train = parse_data_sentence(train)
        dev = parse_data_sentence(dev)
        test = parse_data_sentence(test)
    if RESAMP == 'resamp':
        train = re_sampling(train,M,N)
        dev = re_sampling(dev,M,N)

    # 训练集中提取词表
    word_counts = Counter(row[0].lower() for sample in train for row in sample)
    # 获取词表 {词：词频>2}
    vocab = [w for w, f in iter(word_counts.items()) if f >= 2]

    # save initial config data
    with open(PKL_PATH, 'wb') as outp:
        pickle.dump((vocab, chunk_tags), outp)

    # 已经转换为数字序列
    train = _process_data(train, vocab, chunk_tags)
    dev = _process_data(dev, vocab, chunk_tags)
    test = _process_data(test, vocab, chunk_tags)

    return  train,dev,test

def _parse_data(fh):
    #  in windows the new line is '\r\n\r\n' the space is '\r\n' . so if you use windows system,
    #  you have to use recorsponding instructions

    if platform.system() == 'Windows':
        split_text = '\r\n'
        print(fh,"window")
    else:
        print(fh,'ubuntu')
        split_text = '\r\n'

    string = fh.read().decode('utf-8')
    # 可能有三列
    data = [[row.split() for row in sample.split(split_text)] for
            sample in
            string.strip().split(split_text + split_text) if len(sample.split(split_text))>1]
    print(fh,len(data))
    fh.close()

    return data

def _process_data(data, vocab, chunk_tags, maxlen=None, onehot=False):
    # 处理成数字序列
    if maxlen is None:
        maxlen = max(len(s) for s in data)
    maxlen = 200
    if DIC:
        word2idx = dict((w, i) for i, w in enumerate(vocab))
        x = [[word2idx.get(str(w[0].lower())+str(w[1]), 1) for w in s] for s in data]  # set to <unk> (index 1) if not in vocab
        y_chunk = [[chunk_tags.index(w[2]) for w in s] for s in data]
    else:
        word2idx = dict((w, i) for i, w in enumerate(vocab))
        x = [[word2idx.get(w[0].lower(), 1) for w in s] for s in data]  # set to <unk> (index 1) if not in vocab
        y_chunk = [[chunk_tags.index(w[1]) for w in s] for s in data]

    # 左边自动补〇 padding处理 不足就直接补〇
    x = pad_sequences(x, maxlen)  # left padding
    y_chunk = pad_sequences(y_chunk, maxlen, value=-1)
    if onehot:
        y_chunk = numpy.eye(len(chunk_tags), dtype='float32')[y_chunk]
    else:
        # print('y_chunk_shape:',y_chunk.shape)
        y_chunk = numpy.expand_dims(y_chunk, 2)

    return x, y_chunk

def return_sindex(data):
    indexi = [0]
    for i, row in enumerate(data):
        if row[0] in ['。', '！','；']:
            indexi.append(i)
    return indexi

def parse_data_sentence(tests):
    # 切分句子 最大长度不超过200
    dataset = []

    for test in tests:
        if len(test) < 100 and len(test)>0:
            dataset.append(test)
        else:
            indexi = return_sindex(test)
            # print('i',len(test),test)
            # print(indexi)
            j = 0
            for ii, i in enumerate(indexi):
                if i - j > 90:
                    # dataset.append(test[j:i + 1])
                    if len(test[j:i + 1]) > 200:
                        indexii = return_sindex(test[j:i + 1])
                        data = test[j:i + 1]
                        m = 0
                        for ki, k in enumerate(indexii):
                            if k - m > 100:
                                # 回退机制
                                k = indexii[ki - 1]
                                if k != m:
                                    dataset.append(data[m:k + 1])
                                m = k + 1

                            if ki == len(indexii) - 1:
                                k = indexii[ki]
                                if k!=m:
                                    dataset.append(data[m:k + 1])
                                    break
                    else:
                        if i != j:
                            dataset.append(test[j:i + 1])
                    j = i + 1
                if ii == len(indexi)-1:
                    if i!=j:
                        dataset.append(test[j:i + 1])
    dataset = [data for data in dataset if len(data)>0]
    print('data_sentence:', len(dataset))

    return dataset

def random_shuff(train_x,train_y):
    # 随机打乱训练数据
    tuple_ = []
    for i in range(len(train_x)):
        tuple_.append((train_x[i], train_y[i]))

    random.shuffle(tuple_)
    train_x = []
    train_y = []
    for k, v in tuple_:
        train_x.append(k)
        train_y.append(v)

    train_x = numpy.array(train_x)
    train_y = numpy.array(train_y)
    return train_x,train_y

def random_shuff_k(train_x,train_y,kk):
    # 随机打乱训练数据
    tuple_ = []
    for i in range(len(train_x)):
        tuple_.append((train_x[i], train_y[i]))

    random.shuffle(tuple_)
    train_x = []
    train_y = []
    dev_x = []
    dev_y = []
    i = 1
    for k, v in tuple_:
        if i % kk ==0:
            dev_x.append(k)
            dev_y.append(v)
        else:
            train_x.append(k)
            train_y.append(v)
        i+=1
    train_x = numpy.array(train_x)
    train_y = numpy.array(train_y)
    dev_x = numpy.array(dev_x)
    dev_y = numpy.array(dev_y)

    return train_x,train_y,dev_x,dev_y

def re_sampling(train_data,m,n):
    # 重采样
    train_datas = []
    TYPS = []
    for train in train_data:
        # label.append(train[1])

        train_datas.append(train)
        flag = False
        label = False
        for traini in train:
            if DATA_SET=='ccks' and str(traini[-1][2:]) in ['DISEASE','TREATMENT']:
                flag = True
                if str(traini[-1][2:])=='TREATMENT':
                    label=True
                break
            elif DATA_SET=='ccks2' and str(traini[-1][2:]) in ['OP','DG']:
                flag = True
                if str(traini[-1][2:])=='DG':
                    label=True
                break

        if flag:
            mi=m
            while mi>0:
                train_datas.append(train)
                mi-=1
        if label:
            ni = n
            while ni > 0:
                train_datas.append(train)
                ni -= 1
    print('resampling:',len(train_datas))

    return train_datas




def embedding_mat(vocab,glove_dir):
    # vocab = {}  # 词汇表为数据预处理后得到的词汇字典

    # 构建词向量索引字典
    # 读入词向量文件，文件中的每一行的第一个变量是单词，后面的一串数字对应这个词的词向量
    # glove_dir = "./data/zhwiki_2017_03.sg_50d.word2vec"
    f = open(glove_dir, "r", encoding="utf-8")

    # 获取词向量的维度,l表示单词数，w为某个单词转化为词向量后的维度,
    # 注意，部分预训练好的词向量的第一行并不是该词向量的维度
    l, w = f.readline().split()

    # 创建词向量索引字典
    embeddings_index = {}
    for line in f:
        # 读取词向量文件中的每一行
        values = line.split()
        # 获取当前行的词
        word = values[0]
        # 获取当前词的词向量
        coefs = numpy.asarray(values[1:], dtype="float32")
        # 将读入的这行词向量加入词向量索引字典
        embeddings_index[word] = coefs
    f.close()
    # print(embeddings_index)
    # 构建词向量矩阵，预训练的词向量中没有出现的词用0向量表示
    # 创建一个0矩阵，这个向量矩阵的大小为（词汇表的长度+1，词向量维度）
    embedding_matrix = numpy.zeros((len(vocab) + 1, int(w)))
    # 遍历词汇表中的每一项
    for i,word in enumerate(vocab):
        # 在词向量索引字典中查询单词word的词向量
        embedding_vector = embeddings_index.get(word)
        # 判断查询结果，如果查询结果不为空,用该向量替换0向量矩阵中下标为i的那个向量
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return int(w),embedding_matrix


def embedding_mat2(vocab,glove_dir,ccks,dot):
    # vocab = {}  # 词汇表为数据预处理后得到的词汇字典

    # 构建词向量索引字典
    # 读入词向量文件，文件中的每一行的第一个变量是单词，后面的一串数字对应这个词的词向量
    # glove_dir = "./data/zhwiki_2017_03.sg_50d.word2vec"
    f = open(glove_dir, "r", encoding="utf-8")
    f1 = open(ccks, "r", encoding="utf-8")

    # 获取词向量的维度,l表示单词数，w为某个单词转化为词向量后的维度,
    # 注意，部分预训练好的词向量的第一行并不是该词向量的维度
    l, w = f.readline().split()
    l1, w1 = f1.readline().split()

    # 创建词向量索引字典
    embeddings_index = {}
    for line in f:
        # 读取词向量文件中的每一行
        values = line.split()
        # 获取当前行的词
        word = values[0]
        # 获取当前词的词向量
        coefs = numpy.asarray(values[1:], dtype="float32")
        # 将读入的这行词向量加入词向量索引字典
        embeddings_index[word] = coefs
    f.close()
    embeddings_index2 = {}
    for line2 in f1:
        # 读取词向量文件中的每一行
        values2 = line2.split()
        # 获取当前行的词
        word2 = values2[0]
        # 获取当前词的词向量
        coefs = numpy.asarray(values2[1:], dtype="float32")
        # 将读入的这行词向量加入词向量索引字典
        embeddings_index2[word2] = coefs
    # print(embeddings_index)
    # 构建词向量矩阵，预训练的词向量中没有出现的词用0向量表示
    # 创建一个0矩阵，这个向量矩阵的大小为（词汇表的长度+1，词向量维度）
    if dot==200:
        embedding_matrix = numpy.zeros((len(vocab) + 1, int(w)+int(w1)))
    else:
        embedding_matrix = numpy.zeros((len(vocab) + 1, int(w)))
    # 遍历词汇表中的每一项
    for i,word in enumerate(vocab):
        # 在词向量索引字典中查询单词word的词向量
        embedding_vector = embeddings_index.get(word)
        embedding_vector2 = embeddings_index2.get(word)
        # print(word,embedding_vector2)
        # 判断查询结果，如果查询结果不为空,用该向量替换0向量矩阵中下标为i的那个向量
        if embedding_vector is not None:
            if embedding_vector2 is not None:
                if dot == 200:
                    embedding_matrix[i] = numpy.append(embedding_vector, embedding_vector2)
                #     对应元素相乘
                else:
                    embedding_matrix[i] = numpy.multiply(embedding_vector, embedding_vector2)
            # else:


    return int(w),embedding_matrix

def process_data(data, vocab, maxlen):
    # 测试时用到此函数
    if DIC:
        word2idx = dict((w, i) for i, w in enumerate(vocab))
        # 汉字与标签的组合序列
        # print(data)
        # 匹配词典格式 大小写 ['支B-b', '额E-b', '病E-d']
        x = [word2idx.get(str(w[0].lower())+str(w[1:]), 1) for w in data]
        # print(x)
    else:
        word2idx = dict((w, i) for i, w in enumerate(vocab))
        # 汉字在序列
        x = [word2idx.get(w[0].lower(), 1) for w in data]

    length = len(x)
    x = pad_sequences([x], maxlen)  # left padding
    return x, length

def iob_iobes(tags):
    """
    IOB -> IOBES
    """
    new_tags = []
    for i, tag in enumerate(tags):
        if tag == 'O':
            new_tags.append(tag)
        elif tag.split('-')[0] == 'B':
            if i + 1 != len(tags) and \
               tags[i + 1].split('-')[0] == 'I':
                new_tags.append(tag)
            else:
                new_tags.append(tag.replace('B-', 'S-'))
        elif tag.split('-')[0] == 'I':
            if i + 1 < len(tags) and \
                    tags[i + 1].split('-')[0] == 'I':
                new_tags.append(tag)
            else:
                new_tags.append(tag.replace('I-', 'E-'))
        else:
            raise Exception('Invalid IOB format!')
    return new_tags

def data_iobes(fh,out_ph):
    data = _parse_data(fh)
    f = open(out_ph,'w',encoding='utf-8')
    datas = []
    for test in data:
        strs = []
        label = []
        for row in test:
            strs.append(row[0])
            label.append(row[1])
        label = iob_iobes(label)
        datasi = []
        for text,labeli in zip(strs,label):
            datasi.append([text,labeli])
            f.write(text+' '+labeli+'\n')
        datas.append(datasi)
        f.write('\n')

    return datas

def pre_ccks_label(predict_text, result_tags):
    """
    转换为ccks测评文件形式
    :param predict_text:
    :param result_tags:
    :return:
    """
    i = 1
    be = 0
    mention = []
    bodyi = ''
    typei = ''

    for s, t in zip(predict_text, result_tags):
        if t[0] in ('B', 'I','E','S'):
            if DIC:
                bodyi+=str(s)
            else:
                bodyi += s

            typei += t.split('-')[1] + '-'
            if t[0] == 'B':
                be = i
            else:
                end = i
                if t[0]=='E':
                    # print(bodyi,be,end,typei.split('-')[0])
                    mention.append(bodyi+' '+str(be)+' '+str(end)+' '+typei.split('-')[0])
                    bodyi = ''
                    typei = ''
                if t[0]=='S':
                    # print(bodyi, i, i, typei.split('-')[0])
                    mention.append(bodyi + ' ' + str(i) + ' ' + str(i) + ' ' + typei.split('-')[0])
                    bodyi = ''
                    typei = ''
        i+=1

    # print(mention)
    return mention

def cmp_labels(predict_text,label,result_tags):
    i = 1
    for q,k,v in zip(predict_text,label,result_tags):
        print(i,q,k,v)
        i+=1

def get_dic(iobes_data_path,dic_path,com_path=None,iobes=True):
    """
    接收训练数据
    IOBES标签数据
    """
    print('加载文件：',iobes_data_path)
    data = _parse_data(open(iobes_data_path, 'rb'))
    predict_texts = []
    ccks_labels =[]
    dic = {'BODY':[],'SIGNS':[],'CHECK':[],'DISEASE':[],'TREATMENT':[],}

    for test in data:
        label = []
        predict_text = ''
        for row in test:
            label.append(row[1])
            predict_text += row[0]
        predict_texts.append(predict_text)

        if iobes:
            label = iob_iobes(label)

        m1 = pre_ccks_label(predict_text,label)
        if len(m1)>0:
            for mi in m1:
                mitem = mi.split(' ')
                dic[mitem[-1]].append(mitem[0])

    for d in dic.keys():
        vocab_com = []
        # 训练集中提取词表
        f = open(dic_path+d+'_com_dic.txt','w',encoding='utf-8')
        if com_path!=None and os.path.exists(com_path+d.lower()+'.txt'):
            f1 = open(com_path+d.lower()+'.txt','r',encoding='utf-8')
            for line in f1.readlines():
                vocab_com.append(line.replace('\n',''))
        else:
            print(com_path+d.lower()+'.txt is not exist!')
        word_counts = Counter(row.lower() for row in dic[d])
        print(word_counts)
        # 获取词表 {词：词频>2}
        vocab = [w for w, f in iter(word_counts.items())]
        print(d,len(vocab),vocab)
        for v in vocab:
            f.write(v+'\n')
        for v in vocab_com:
            if v not in vocab:
                f.write(v+'\n')

        print(d,len(dic[d]),dic[d])
        # dic[d] = list(set(dic[d]))
        # print(d, len(dic[d]), dic[d])

        # print(predict_text)
        # print(m1)
        # ccks_labels.append(m1)

        # print(m1)
    return dic

if __name__ == "__main__":
    pass
    # (train_x, train_y), (test_x, test_y), (vocab, chunk_tags) = load_data()
    # print('vocab:',vocab)
    # print(test_x)
    # w,embedding_matrix = embedding_mat(vocab, 'emb/wiki_100_1.utf8')
    # print(embedding_matrix)

    # data_iobes(open('data/data/train_dev_set.txt', 'rb'),'data/data/train_dev_iobes_set.txt')
    # data_iobes(open('data/data/test_set.txt', 'rb'), 'data/data/test_iobes_set.txt')
    # """
    TRAIN_PAH = DATA_PATH + 'train_dev_iobes_set.txt'
    train = _parse_data(open(TRAIN_PAH, 'rb'))
    data = parse_data_sentence(train)
    data = re_sampling(data)
    #
    """
    maxlen = max(len(s) for s in data)
    lens = [s for s in data if len(s)<=50]
    lens0 = [s for s in data if len(s)<=100 and len(s)>50]
    lens1 = [s for s in data if len(s)>100 and len(s)<=200]
    lens2 = [s for s in data if len(s)>200]
    print('all:',len(data)) # 2033
    print('<50:',len(lens)) # 144<50
    print('50-100:',len(lens0)) # 144<50
    print('100-200:',len(lens1)) # 1191 1156
    print('200>',len(lens2)) # 31
    lens10 = [s for s in data if len(s)==100]
    print('100:',len(lens10))
    #
    # print(maxlen)
    #
    """

    # tr = [['患', 'O'], ['者', 'O'], ['自', 'O'], ['入', 'O'], ['院', 'O'], ['以', 'O'], ['来', 'O'], ['，', 'O'], ['无', 'O'], ['发', 'B-SIGNS'], ['热', 'I-SIGNS'], ['，', 'O'], ['无', 'O'], ['头', 'B-SIGNS'], ['晕', 'I-SIGNS'], ['头', 'B-SIGNS'], ['痛', 'I-SIGNS'], ['，', 'O'], ['无', 'O'], ['恶', 'B-SIGNS'], ['心', 'I-SIGNS'], ['呕', 'B-SIGNS'], ['吐', 'I-SIGNS'], ['，', 'O'], ['无', 'O'], ['胸', 'B-SIGNS'], ['闷', 'I-SIGNS'], ['心', 'B-SIGNS'], ['悸', 'I-SIGNS'], ['，', 'O'], ['无', 'O'], ['腹', 'B-SIGNS'], ['胀', 'I-SIGNS'], ['腹', 'B-SIGNS'], ['痛', 'I-SIGNS'], ['，', 'O'], ['饮', 'O'], ['食', 'O'], ['可', 'O'], ['，', 'O'], ['大', 'B-BODY'], ['小', 'I-BODY'], ['便', 'I-BODY'], ['均', 'O'], ['正', 'O'], ['常', 'O'], ['。', 'O'], ['4', 'O'], ['.', 'O'], ['查', 'B-CHECK'], ['体', 'I-CHECK'], ['：', 'O'], ['T', 'B-CHECK'], ['3', 'O'], ['6', 'O'], ['.', 'O'], ['7', 'O'], ['C', 'O'], ['，', 'O'], ['P', 'B-CHECK'], ['7', 'O'], ['8', 'O'], ['次', 'O'], ['/', 'O'], ['分', 'O'], ['，', 'O'], ['R', 'B-CHECK'], ['1', 'O'], ['8', 'O'], ['次', 'O'], ['/', 'O'], ['分', 'O'], ['，', 'O'], ['B', 'B-CHECK'], ['P', 'I-CHECK'], ['1', 'O'], ['2', 'O'], ['0', 'O'], ['/', 'O'], ['8', 'O'], ['0', 'O'], ['m', 'O'], ['m', 'O'], ['H', 'O'], ['g', 'O'], [',', 'O'], ['心', 'B-CHECK'], ['肺', 'I-CHECK'], ['腹', 'I-CHECK'], ['查', 'I-CHECK'], ['体', 'I-CHECK'], ['未', 'O'], ['见', 'O'], ['明', 'O'], ['显', 'O'], ['异', 'O'], ['常', 'O'], ['，', 'O'], ['专', 'O'], ['科', 'O'], ['情', 'O'], ['况', 'O'], ['：', 'O'], ['头', 'B-BODY'], ['顶', 'I-BODY'], ['部', 'I-BODY'], ['广', 'O'], ['泛', 'O'], ['触', 'B-CHECK'], ['痛', 'I-CHECK'], ['，', 'O'], ['无', 'O'], ['明', 'O'], ['显', 'O'], ['肿', 'B-SIGNS'], ['胀', 'I-SIGNS'], ['，', 'O'], ['双', 'B-BODY'], ['侧', 'I-BODY'], ['瞳', 'I-BODY'], ['孔', 'I-BODY'], ['正', 'B-SIGNS'], ['大', 'I-SIGNS'], ['等', 'I-SIGNS'], ['圆', 'I-SIGNS'], ['，', 'O'], ['光', 'B-CHECK'], ['反', 'I-CHECK'], ['射', 'I-CHECK'], ['灵', 'O'], ['敏', 'O'], ['，', 'O'], ['右', 'B-BODY'], ['前', 'I-BODY'], ['额', 'I-BODY'], ['肿', 'B-SIGNS'], ['胀', 'I-SIGNS'], ['，', 'O'], ['压', 'B-CHECK'], ['痛', 'I-CHECK'], ['，', 'O'], ['下', 'B-BODY'], ['颌', 'I-BODY'], ['部', 'I-BODY'], ['可', 'O'], ['见', 'O'], ['片', 'O'], ['状', 'O'], ['挫', 'O'], ['伤', 'O'], ['，', 'O'], ['渗', 'O'], ['血', 'O'], ['，', 'O'], ['压', 'B-CHECK'], ['痛', 'I-CHECK'], ['，', 'O'], ['颈', 'O'], ['软', 'O'], ['，', 'O'], ['无', 'O'], ['抵', 'O'], ['抗', 'O'], ['，', 'O'], ['双', 'B-BODY'], ['臀', 'I-BODY'], ['部', 'I-BODY'], ['肿', 'B-SIGNS'], ['胀', 'I-SIGNS'], ['，', 'O'], ['压', 'B-CHECK'], ['痛', 'I-CHECK'], ['，', 'O'], ['双', 'B-BODY'], ['侧', 'I-BODY'], ['髋', 'I-BODY'], ['部', 'I-BODY'], ['无', 'O'], ['明', 'O'], ['显', 'O'], ['肿', 'B-SIGNS'], ['胀', 'I-SIGNS'], ['，', 'O'], ['左', 'B-BODY'], ['侧', 'I-BODY'], ['髋', 'I-BODY'], ['关', 'I-BODY'], ['节', 'I-BODY'], ['活', 'O'], ['动', 'O'], ['可', 'O'], ['，', 'O'], ['右', 'B-BODY'], ['侧', 'I-BODY'], ['髋', 'I-BODY'], ['关', 'I-BODY'], ['节', 'I-BODY'], ['活', 'O'], ['动', 'O'], ['受', 'O'], ['限', 'O'], ['，', 'O'], ['骨', 'B-BODY'], ['盆', 'I-BODY'], ['分', 'O'], ['离', 'O'], ['挤', 'O'], ['压', 'O'], ['实', 'O'], ['验', 'O'], ['（', 'O'], ['）', 'O'], ['，', 'O'], ['右', 'B-BODY'], ['大', 'I-BODY'], ['腿', 'I-BODY'], ['、', 'O'], ['右', 'B-BODY'], ['膝', 'I-BODY'], ['部', 'I-BODY'], ['及', 'O'], ['右', 'B-BODY'], ['小', 'I-BODY'], ['腿', 'I-BODY'], ['广', 'O'], ['泛', 'O'], ['压', 'B-CHECK'], ['痛', 'I-CHECK'], ['，', 'O'], ['叩', 'B-CHECK'], ['击', 'I-CHECK'], ['痛', 'I-CHECK'], ['阳', 'O'], ['性', 'O'], ['，', 'O'], ['未', 'O'], ['触', 'O'], ['及', 'O'], ['骨', 'O'], ['擦', 'O'], ['感', 'O'], ['，', 'O'], ['右', 'B-BODY'], ['膝', 'I-BODY'], ['、', 'I-BODY'], ['右', 'I-BODY'], ['踝', 'I-BODY'], ['关', 'I-BODY'], ['节', 'I-BODY'], ['活', 'O'], ['动', 'O'], ['障', 'O'], ['碍', 'O'], ['，', 'O'], ['右', 'B-BODY'], ['足', 'I-BODY'], ['背', 'I-BODY'], ['动', 'I-BODY'], ['脉', 'I-BODY'], ['搏', 'O'], ['动', 'O'], ['好', 'O'], ['，', 'O'], ['足', 'B-BODY'], ['趾', 'I-BODY'], ['感', 'O'], ['觉', 'O'], ['运', 'O'], ['动', 'O'], ['正', 'O'], ['常', 'O'], ['。', 'O']]
    # print(str(tr[1][2:]))

    # dic = get_dic('data/train_set.txt','data/dic/','data/dict/')
    # print(dic)