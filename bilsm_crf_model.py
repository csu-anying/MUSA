# coding:utf-8

import keras
from keras.models import Sequential
from keras.layers import Embedding, Dropout, Conv1D, Bidirectional, GRU, LSTM, ZeroPadding1D, TimeDistributed, Input, \
    Dense, BatchNormalization, Masking, Softmax
from keras.models import Model, load_model
from keras.layers.merge import concatenate, multiply
from keras.utils import plot_model, multi_gpu_model
from keras.optimizers import Adam, SGD

from keras_contrib.layers import CRF
from keras_contrib.losses import crf_loss
from keras_contrib.metrics import crf_viterbi_accuracy, crf_accuracy
from keras_self_attention import SeqSelfAttention
from tensorflow.python.keras.backend import softmax

from keras_multi_head import MultiHead, MultiHeadAttention
import process_data
import pickle
from model_train import Dropout_r, LSTM_UNIT, DIC, EMBED_DIMs, DATA_SET, LTYPE, CRF_Flag
from parallel_model import ParallelModel

EMBED_DIM = EMBED_DIMs
LSTM_UNITS = LSTM_UNIT
half_window_size = 2
dropout_r = Dropout_r


# 此文件为模型的具体实现，需要搭配论文一起看。

# 一个消融模型
def Bilstm_Crf(maxlen, char_value_dict_len, class_label_count, embedding_weights=None, is_train=True):
    # 输入层
    word_input = Input(shape=(maxlen,), dtype='int32', name='word_input')
    print('\n\n\n*****************************,词典大小：', char_value_dict_len + 1,
          '**************************************\n\n\n')
    if is_train:
        # mask_zero=True,
        word_emb = Embedding(char_value_dict_len + 1, output_dim=EMBED_DIM, mask_zero=True,
                             input_length=maxlen, weights=[embedding_weights],
                             name='word_emb')(word_input)
    else:
        word_emb = Embedding(char_value_dict_len + 1, output_dim=EMBED_DIM,
                             input_length=maxlen,
                             name='word_emb')(word_input)

    bilstm = Bidirectional(LSTM(LSTM_UNITS, return_sequences=True))(word_emb)
    bilstm_d = Dropout(dropout_r)(bilstm)
    bilstm_d = BatchNormalization()(bilstm_d)
    dense = TimeDistributed(Dense(class_label_count))(bilstm_d)
    if CRF_Flag:
        crf = CRF(class_label_count, sparse_target=True)
        crf_output = crf(dense)
    else:
        crf_output = Softmax()(dense)

    model = Model(inputs=[word_input], outputs=[crf_output])

    optimizer = Adam(lr=0.01, clipvalue=0.5)
    if CRF_Flag:
        model.compile(loss=crf_loss, optimizer=optimizer, metrics=[crf_viterbi_accuracy])
    else:
        model.compile(optimizer=optimizer, loss='categorical_crossentropy')
    model.summary()

    return model


# 最终的模型
def Bilstm_self_Crf(maxlen, char_value_dict_len, class_label_count, embedding_weights=None, is_train=True):
    # 输入层
    word_input = Input(shape=(maxlen,), dtype='int32', name='word_input')
    # Embedding层
    if is_train:
        word_emb = Embedding(char_value_dict_len + 1, output_dim=EMBED_DIM,
                             input_length=maxlen, weights=[embedding_weights], mask_zero=True,
                             name='word_emb')(word_input)
    else:
        word_emb = Embedding(char_value_dict_len + 1, output_dim=EMBED_DIM,
                             input_length=maxlen,
                             name='word_emb')(word_input)
    # 多头注意力
    word_emb1 = MultiHeadAttention(head_num=EMBED_DIM, name='Multi-Head')(word_emb)
    word_emb1 = Dropout(dropout_r)(word_emb1)
    word_emb1 = BatchNormalization()(word_emb1)

    # bilstm
    bilstm = Bidirectional(LSTM(LSTM_UNITS, return_sequences=True))(word_emb)
    bilstm_d = Dropout(dropout_r)(bilstm)
    bilstm_d = BatchNormalization()(bilstm_d)

    # 将两者拼接，输入一个线性层后输出
    self_self = concatenate([bilstm_d, word_emb1], axis=2)
    dense = TimeDistributed(Dense(class_label_count))(self_self)
    # crf
    crf = CRF(class_label_count, sparse_target=True)
    crf_output = crf(dense)

    # build model
    model = Model(inputs=[word_input], outputs=[crf_output])
    # 控制梯度裁剪 梯度被裁剪到-0.5~0.5之间
    optimizer = Adam(lr=0.01, clipvalue=0.5)
    model.compile(optimizer=optimizer, loss=crf_loss, metrics=[crf_viterbi_accuracy])
    model.summary()

    return model


def create_model_base_(maxlen, train=True):
    if DIC:
        if train:
            (train_x, train_y), (test_x, test_y), (vocab, vocab0, vocab1, chunk_tags) = process_data.load_data()
        else:
            with open(process_data.PKL_PATH, 'rb') as inp:
                (vocab, vocab0, vocab1, chunk_tags) = pickle.load(inp)
        # word word+label
        #         EMBED_DIM, embedding_matrix = process_data.embedding_mat(vocab1, './emb/ldic_feature100.txt')
        if DATA_SET == 'ccks':
            if LTYPE == 'l':
                print('********************lllllllllllllllllllllll*************')
                EMBED_DIM, embedding_matrix = process_data.dembedding_mat(vocab0, vocab1, './emb/embedding_B2.txt',
                                                                          './emb/wiki_100_1.utf8', EMBED_DIMs, LTYPE)
            else:
                EMBED_DIM, embedding_matrix = process_data.dembedding_mat(vocab0, vocab1, './emb/embedding_C2.txt',
                                                                          './emb/wiki_100_1.utf8', EMBED_DIMs, LTYPE)
        elif DATA_SET == 'ccks2':
            if LTYPE == 'l':
                EMBED_DIM, embedding_matrix = process_data.dembedding_mat(vocab0, vocab1, './emb/embedding_B.txt',
                                                                          './emb/wiki_100_1.utf8', EMBED_DIMs, LTYPE)
            else:
                EMBED_DIM, embedding_matrix = process_data.dembedding_mat(vocab0, vocab1, './emb/embedding_C.txt',
                                                                          './emb/wiki_100_1.utf8', EMBED_DIMs, LTYPE)
            # EMBED_DIM, embedding_matrix = process_data.dembedding_mat(vocab0, vocab1, './emb/embedding_C.txt',
            #                                                           './emb/wiki_100_1.utf8', EMBED_DIMs)
    else:
        if train:
            (train_x, train_y), (test_x, test_y), (vocab, chunk_tags) = process_data.load_data()
        else:
            with open(process_data.PKL_PATH, 'rb') as inp:
                (vocab, chunk_tags) = pickle.load(inp)
        # EMBED_DIM, embedding_matrix = process_data.embedding_mat(vocab, 'emb/ccks_emb.txt')
        EMBED_DIM, embedding_matrix = process_data.embedding_mat(vocab, 'emb/wiki_100_1.utf8')
        # EMBED_DIM, embedding_matrix = process_data.embedding_mat2(vocab, 'emb/wiki_100_1.utf8','emb/ccks_emb.txt',200)

    print('vocab:', len(vocab))
    print('chunk_len:', len(chunk_tags))
    print('em_dim:', EMBED_DIM)

    model = Bilstm_Crf(maxlen, len(vocab), len(chunk_tags), embedding_weights=embedding_matrix)
    # plot_model(model,to_file='bilstm_cnn_crf_model.png',show_shapes=True,show_layer_names=True)
    if train:
        return model, (train_x, train_y), (test_x, test_y)
    else:
        if DIC:
            return model, (vocab, vocab0, vocab1, chunk_tags)
        else:
            return model, (vocab, chunk_tags)


def create_model_self(maxlen, train=True):
    if DIC:
        if train:
            (train_x, train_y), (test_x, test_y), (vocab, vocab0, vocab1, chunk_tags) = process_data.load_data()
        else:
            print(process_data.PKL_PATH)
            with open(process_data.PKL_PATH, 'rb') as inp:
                (vocab, vocab0, vocab1, chunk_tags) = pickle.load(inp)
        if DATA_SET == 'ccks':
            EMBED_DIM, embedding_matrix = process_data.dembedding_mat(vocab0, vocab1, './emb/ldic_feature100.txt',
                                                                      './emb/wiki_100_1.utf8', EMBED_DIMs, 'l')
    #             if LTYPE=='l':
    #                 print('***************lllllll*********************')
    #                 EMBED_DIM, embedding_matrix = process_data.dembedding_mat(vocab0, vocab1, './emb/embedding_B2.txt',
    #                                                                           './emb/wiki_100_1.utf8', EMBED_DIMs, LTYPE)
    #             else:
    #                 EMBED_DIM, embedding_matrix = process_data.dembedding_mat(vocab0, vocab1, './emb/embedding_C2.txt',
    #                                                                   './emb/wiki_100_1.utf8',EMBED_DIMs ,LTYPE)
    #         elif DATA_SET=='ccks2':
    #             if LTYPE=='l':
    #                 EMBED_DIM, embedding_matrix = process_data.dembedding_mat(vocab0, vocab1, './emb/embedding_B.txt',
    #                                                                           './emb/wiki_100_1.utf8', EMBED_DIMs, LTYPE)
    #             else:
    #                 EMBED_DIM, embedding_matrix = process_data.dembedding_mat(vocab0, vocab1, './emb/embedding_C.txt',
    #                                                                   './emb/wiki_100_1.utf8',EMBED_DIMs ,LTYPE)
    else:
        if train:
            (train_x, train_y), (test_x, test_y), (vocab, chunk_tags) = process_data.load_data()
        else:
            with open(process_data.PKL_PATH, 'rb') as inp:
                (vocab, chunk_tags) = pickle.load(inp)
        # EMBED_DIM, embedding_matrix = process_data.embedding_mat(vocab, 'emb/ccks_emb.txt')
        EMBED_DIM, embedding_matrix = process_data.embedding_mat(vocab, 'emb/wiki_100_1.utf8')
        # EMBED_DIM, embedding_matrix = process_data.embedding_mat2(vocab, 'emb/wiki_100_1.utf8', 'emb/ccks_emb.txt', 100)

    print('vocab:', len(vocab))
    print('chunk_len:', len(chunk_tags))
    print('em_dim:', EMBED_DIM)
    print("Embedding_matrix.shape:", embedding_matrix.shape)

    model = Bilstm_self_Crf(maxlen, len(vocab), len(chunk_tags), embedding_weights=embedding_matrix)
    # plot_model(model,to_file='bilstm_cnn_crf_model.png',show_shapes=True,show_layer_names=True)

    if train:
        return model, (train_x, train_y), (test_x, test_y)
    else:
        if DIC:
            return model, (vocab, vocab0, vocab1, chunk_tags)
        else:
            return model, (vocab, chunk_tags)
