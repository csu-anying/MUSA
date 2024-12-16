# coding:utf-8
from gensim.models import Word2Vec
import numpy


def get_Sentences(data_path):
    f = open(data_path, 'r', encoding='utf-8')
    lines = f.readlines()
    sentences = []
    words = []
    word = []
    sentence = []
    for line in lines:
        line = line.replace('\n', '')
        # print(line)
        if line != '':
            # print("**")
            # 停用词去除
            w = line.split(' ')[0]
            dl = line.split(' ')[1]
            word.append(line.split(' ')[0])
            sentence.append(str(w) + str(dl))
        else:
            # print(sentence)
            words.append(word)
            sentences.append(sentence)
            sentence = []
            word = []
    return words, sentences


# print(len(sentences))
# print(sentences[1])
def dembedding_mat(vocab, vocab1, dic_emb, glove_dir, dot, ltype):
    # vocab = {}  # 词汇表为数据预处理后得到的词汇字典

    # 构建词向量索引字典
    # 读入词向量文件，文件中的每一行的第一个变量是单词，后面的一串数字对应这个词的词向量
    # glove_dir = "./data/zhwiki_2017_03.sg_50d.word2vec"
    f = open(glove_dir, "r", encoding="utf-8")
    fd = open(dic_emb, "r", encoding="utf-8")

    # 获取词向量的维度,l表示单词数，w为某个单词转化为词向量后的维度,
    # 注意，部分预训练好的词向量的第一行并不是该词向量的维度
    l, w = f.readline().split()
    ld, wd = fd.readline().split()

    # 创建词向量索引字典
    embeddings_index = {}
    dembeddings_index = {}
    for line in f:
        # 读取词向量文件中的每一行
        values = line.split()
        # 获取当前行的词
        word = values[0]
        # 获取当前词的词向量
        coefs = numpy.asarray(values[1:], dtype="float32")
        # 将读入的这行词向量加入词向量索引字典
        embeddings_index[word] = coefs
    for line in fd:
        # 读取词向量文件中的每一行
        values = line.split()
        # 获取当前行的词
        word = values[0]
        # 获取当前词的词向量
        coefs = numpy.asarray(values[1:], dtype="float32")
        # 将读入的这行词向量加入词向量索引字典
        dembeddings_index[word] = coefs
    fd.close()
    f.close()
    # print(embeddings_index)
    # 构建词向量矩阵，预训练的词向量中没有出现的词用0向量表示
    # 创建一个0矩阵，这个向量矩阵的大小为（词汇表的长度+1，词向量维度）
    if dot == 200:
        embedding_matrix = numpy.zeros((len(vocab) + 1, int(w) + int(wd)))
    else:
        embedding_matrix = numpy.zeros((len(vocab) + 1, int(w)))
    # 遍历词汇表中的每一项
    i = 0
    for word, dicl in zip(vocab, vocab1):
        # 在词向量索引字典中查询单词word的词向量
        embedding_vector = embeddings_index.get(word)
        if ltype == 'l':
            dembedding_vector = dembeddings_index.get(dicl[-1])
        else:
            dembedding_vector = dembeddings_index.get(dicl)
        # print(dicl,word)
        # print(dicl,dembedding_vector)
        # print(word,embedding_vector)
        # 判断查询结果，如果查询结果不为空,用该向量替换0向量矩阵中下标为i的那个向量
        if embedding_vector is not None:
            # 向量拼接 前后拼接组合
            if dot == 200:
                if dembedding_vector is not None:
                    embedding_matrix[i] = numpy.append(embedding_vector, dembedding_vector)
                else:
                    embedding_matrix[i] = numpy.append(embedding_vector, embedding_vector)

            #     对应元素相乘
            else:
                if dembedding_vector is not None:
                    embedding_matrix[i] = numpy.multiply(embedding_vector, dembedding_vector)
                else:
                    embedding_matrix[i] = numpy.multiply(embedding_vector, embedding_vector)
                # embedding_matrix[i] = numpy.multiply(embedding_vector, dembedding_vector)
        else:
            if dot == 200:
                if dembedding_vector is not None:
                    embedding_matrix[i] = numpy.append(dembedding_vector, dembedding_vector)
            else:
                if dembedding_vector is not None:
                    embedding_matrix[i] = numpy.multiply(dembedding_vector, dembedding_vector)

        i += 1
        # assert 1==0
    print(embedding_matrix.shape)
    if dot == 100:
        return int(w), embedding_matrix
    else:
        return int(w) + int(wd), embedding_matrix


def embedding_mat2(vocab, glove_dir, ccks, dot):
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
    embedding_matrix = numpy.zeros((len(vocab) + 1, int(w)))
    # 遍历词汇表中的每一项
    for i, word in enumerate(vocab):
        # 在词向量索引字典中查询单词word的词向量
        embedding_vector = embeddings_index.get(word)
        embedding_vector2 = embeddings_index2.get(word)
        print(word, embedding_vector2)
        # 判断查询结果，如果查询结果不为空,用该向量替换0向量矩阵中下标为i的那个向量
        if embedding_vector is not None:
            if embedding_vector2 is not None:
                if dot == 200:
                    embedding_matrix[i] = numpy.append(embedding_vector, embedding_vector2)
                #     对应元素相乘
                else:
                    embedding_matrix[i] = numpy.multiply(embedding_vector, embedding_vector2)
            else:
                print(word, ' not in ccks!')
        else:
            print(word, ' not in wiki')

    return int(w), embedding_matrix


"""
sg=1 skip-gram
size 输出维度
window 窗口大小
min_count 忽略低于此频率的词
workers 使用的线程数

"""


def train(file_path, model_path, emb_path, emb_dim):
    data_path = './data/ccks/' + file_path
    words, sentences = get_Sentences(data_path)

    model = Word2Vec(sentences=sentences, sg=1, size=emb_dim, window=5, min_count=1, negative=3, sample=0.001, hs=1,
                     workers=4)
    model.save(model_path)
    model = Word2Vec.load(model_path)
    model.wv.save_word2vec_format(emb_path, binary=False)

    # vector = model['B-b']
    # print(vector)
    # word_vectors = model.wv


from collections import Counter
import platform


def dic_cnt(file_path, DATA_SET):
    if DATA_SET == 'ccks':
        data_path = './data/ccks/' + file_path
    elif DATA_SET == 'ccks2':
        data_path = './data/ccks2/' + file_path
    words, sentences = get_Sentences(data_path)
    # 训练集中提取词表
    word_counts = Counter(
        (wordi, sentencei) for word, sentence in zip(words, sentences) for wordi, sentencei in zip(word, sentence))
    # print(word_counts)
    # 获取词表 {词：词频>2}
    vocab = [w for w, f in iter(word_counts.items())]
    vocab0 = [w[0].lower() for w, f in iter(word_counts.items())]
    vocab1 = [str(w[1]) for w, f in iter(word_counts.items())]
    # vocab1 = [w for w, f in iter(word_counts.items()) if w[1] == 'B-c']
    # print(vocab)
    # print(vocab0)
    # print(vocab1)
    return vocab, vocab0, vocab1


def _parse_data(fh):
    #  in windows the new line is '\r\n\r\n' the space is '\r\n' . so if you use windows system,
    #  you have to use recorsponding instructions

    if platform.system() == 'Windows':
        split_text = '\r\n'
        print(fh, "window")
    else:
        split_text = '\n'

    string = fh.read()
    print(string)
    # 可能有三列
    data = [[row.split() for row in sample.split(split_text)] for
            sample in
            string.strip().split(split_text + split_text) if len(sample.split(split_text)) > 1]
    print(fh, len(data))
    fh.close()

    return data


def get_dic(file_path):
    data_path = './data/ccks/' + file_path
    traind = _parse_data(open(data_path, 'r', encoding='utf-8'))
    # 训练集中提取词表
    word_counts = Counter(row[0].lower() for sample in traind for row in sample)
    # 获取词表 {词：词频>2}
    vocab = [w for w, f in iter(word_counts.items()) if f >= 2]
    return vocab


if __name__ == '__main__':
    # 生成向量
    # train('train_dic_set2.txt','./emb/ldic_feature100.model','./emb/ldic_feature100.txt',100)
    """
    vocab,vocab0,vocab1 = dic_cnt('train_dic_set2.txt')
    # vocab,vocab0,vocab1 = dic_cnt('train_dic_set2.txt')
    print(vocab0)
    print(len(vocab1))
    print(vocab1)

    dim,emb = dembedding_mat(vocab0,vocab1,'./emb/ldic_feature100.txt','./emb/wiki_100_1.utf8')
    print(dim)
    print(len(emb))
    model = Word2Vec.load('./emb/ldic_feature100.model')
    s = model.most_similar(positive=['酸I-c','胰B-t'])
    print(s)
    print(vocab[1])
    print(emb[1])
"""
    # vocab1 = get_dic('train_dev_iobes_set.txt')
    # dim, emb = embedding_mat2(vocab1,'./data/emb/wiki_100_1.utf8','./data/emb/ccks_emb.txt',200)
    vocab, vocab1, vocab2 = dic_cnt('train_dic_set2_.txt', 'ccks2')
    print(vocab)
    print(vocab1)
    print(vocab2)
    for v in vocab2:
        print(v[-1])
