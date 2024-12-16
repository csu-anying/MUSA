# coding:utf-8
import argparse
import datetime
import os
import time

# from keras.utils import multi_gpu_model
# import keras
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, CSVLogger, ReduceLROnPlateau, Callback

import bilsm_crf_model
import ccks_eval
import process_data
import val
import os


os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# 此文件用于训练（测试）模型，模型的具体实现在bilsm_crf_model.py文件。
class LossHistory(Callback):
    """
    记录损失历史
    """

    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))


parser = argparse.ArgumentParser()

# 选择Dataset，有两个数据集，CCKS17和CCKS18，默认是17。
parser.add_argument('--dataset', action='store', dest='dataset', default='ccks', type=str, help='Dataset')
# 选择是否有验证集，默认有。
parser.add_argument('--dev', action='store', dest='dev', default='dev_train', type=str, help='dev')
# 选择是否分局，默认不分
parser.add_argument('--tosentence', action='store', dest='tosentence', default=False, type=bool, help='tosentence')
# 没懂什么意思
parser.add_argument('--resamp', action='store', dest='resamp', default='no', type=str, help='resamp')
parser.add_argument('--tagsche', action='store', dest='tagsche', default='iobes', type=str, help='tagsche')
# 是否使用字典，默认使用
parser.add_argument('--dic', action='store', dest='dic', default=True, type=bool, help='dic')
# 此参数应是用于交叉验证，代码中没有使用
parser.add_argument('--K', action='store', dest='k', default=10, type=int, help='k')
# 维度
parser.add_argument('--emb', action='store', dest='emb', default=200, type=int, help='emb')
# 是否使用类型
parser.add_argument('--ltype', action='store', dest='ltype', default='comb', type=str, help='ltype')
# 选择模型，self是其最终模型
parser.add_argument('--usemodel', action='store', dest='usemodel', default='self', type=str, help='usemodel')
# 训练还是测试
parser.add_argument('--mode', action='store', dest='mode', default='train', type=str, help='mode')
# 参数设置
parser.add_argument('--lstm', action='store', dest='lstm', default=128, type=int, help='lstm')
parser.add_argument('--dropout_r', action='store', dest='dropout_r', default=0.1, type=float, help='dropout_r')
parser.add_argument('--split', action='store', dest='split', default=0.1, type=float, help='split')
parser.add_argument('--batchsize', action='store', dest='batchsize', default=16, type=int, help='batchsize')
parser.add_argument('--epoches', action='store', dest='epoches', default=20, type=int, help='epoches')
# 没理解是干什么的
parser.add_argument('--other', action='store', dest='other', default='', type=str, help='other')
# parser.add_argument('--gpu',action='store',dest = 'gpu',default='1',type=str,help='gpu')
# parser.add_argument('--ngpu',action='store',dest = 'ngpu',default=4,type=int,help='ngpu')

opt = parser.parse_args()
# parameters = OrderedDict()
# os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
# NGPU = opt.ngpu
"""
参数设置
"""
DATA_SET = opt.dataset
IOBES = opt.tagsche
DIC = opt.dic
K = opt.k
EMBED_DIMs = opt.emb
LTYPE = opt.ltype

RESAMP = opt.resamp
DEV_TRAIN = opt.dev
TOSENTENCE = opt.tosentence
EPOCHS = opt.epoches
BATCH_SIZE = opt.batchsize
MODEL = opt.usemodel
Dropout_r = opt.dropout_r
SPLIT = opt.split
LSTM_UNIT = opt.lstm
other = opt.other
MODE = opt.mode
CRF_Flag = True
"""
固定结果路径
"""
step = DEV_TRAIN + IOBES + RESAMP
if DIC:
    RESULT_PATH = './result/' + DATA_SET + '/' + step + '/' + MODEL + '_dic' + other + '/'
else:
    RESULT_PATH = './result/' + DATA_SET + '/' + step + '/' + MODEL + '_' + other + '/'
if not os.path.exists('./result/' + DATA_SET + '/' + step + '/'):
    os.mkdir('./result/' + DATA_SET + '/' + step + '/')

if not os.path.exists(RESULT_PATH):
    os.mkdir(RESULT_PATH)

# 读取bio标签数据
PKL_PATH = RESULT_PATH + 'config.pkl'
MODEL_PATH = RESULT_PATH + 'crf.h5'
MODEL_PATH1 = RESULT_PATH + 'crf_step/'
if not os.path.exists(MODEL_PATH1):
    os.mkdir(MODEL_PATH1)
# 返回bioes标签结果
PRE_PATH = RESULT_PATH + 'pre_text.txt'
SCORE_PATH = RESULT_PATH + 'score.txt'
TRAIN_LOG = RESULT_PATH + 'training.log'


def train():
    # plotter = AccLossPlotter(save_path=RESULT_PATH,graphs=['loss'],save_graph=True)
    tensorboad = TensorBoard(log_dir='log')
    # 当验证集的损失的误差不再下降，中断训练  验证集是输入的后 x% patience 间隔n个epoch
    # early_stopping = EarlyStopping(monitor='val_loss',mode='auto', patience=3, verbose=0)
    check_pointer = ModelCheckpoint(filepath=MODEL_PATH1 + '{epoch:02d}-{val_loss:.2f}.h5',
                                    verbose=1, save_best_only=True)
    history = LossHistory()
    csv_logger = CSVLogger(TRAIN_LOG)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=3, mode='auto')
    # call_list = [tensorboad,early_stopping,check_pointer,plotter]
    # call_list = [tensorboad,check_pointer,plotter,csv_logger,reduce_lr]
    call_list = [tensorboad, check_pointer, csv_logger, reduce_lr]

    if DEV_TRAIN != 'dev_train':
        (train_x, train_y), (dev_x, dev_y), (test_x, test_y) = process_data.load_dev_data()
        maxl = max(max(len(s) for s in train_x), max(len(s) for s in dev_x), max(len(s) for s in test_x))
        print(maxl)
    else:
        if DATA_SET == 'ccks':
            maxl = 200
        else:
            (train_x, train_y), (test_x, test_y), dev_x = process_data.load_data()
            maxl = max(max(len(s) for s in train_x), max(len(s) for s in test_x))
            print(maxl)

    model_type = MODEL
    if model_type == 'base_':
        print('*' * 20, 'bilstm_crf_base_', '*' * 20)
        model, (train_x, train_y), (test_x, test_y) = bilsm_crf_model.create_model_base()
    elif model_type == 'self_':
        print('*' * 20, 'bilstm_crf_self_', '*' * 20)
        model, (train_x, train_y), (test_x, test_y) = bilsm_crf_model.create_model_att()
    elif model_type == 'base':
        print('*' * 20, 'bilstm_crf_base', '*' * 20)
        model, (train_x, train_y), (test_x, test_y) = bilsm_crf_model.create_model_base_(maxlen=maxl)
    elif model_type == 'self':
        print('*' * 20, 'self_bilstm_crf', '*' * 20)
        model, (train_x, train_y), (test_x, test_y) = bilsm_crf_model.create_model_self(maxlen=maxl)
    elif model_type == 'mul_self':
        print('*' * 20, 'bilstm_test', '*' * 20)
        model, (train_x, train_y), (test_x, test_y) = bilsm_crf_model.create_model_mul_self(maxlen=maxl)

    # 手动shuffer
    train_x, train_y = process_data.random_shuff(train_x, train_y)
    # train_x, train_y, dev_xx,dev_yy = process_data.random_shuff_k(train_x, train_y,K)

    # model, (train_x, train_y), (test_x, test_y) = bilsm_crf_model.create_model_cnn()
    # train model  validation_data=[dev_x, dev_y] shuffle 自动打乱训练集  先split后shuffer
    # train_y = to_categorical(train_y,19)
    print('*************************', train_y.shape, '****************')
    if DEV_TRAIN == 'dev_train':
        hist = model.fit(train_x, train_y, batch_size=BATCH_SIZE, epochs=EPOCHS, shuffle=True, validation_split=SPLIT,
                         callbacks=call_list)
        # hist = model.fit(train_x, train_y, batch_size=BATCH_SIZE, epochs=EPOCHS, shuffle=True, callbacks=call_list,
        #                  validation_data=[dev_xx, dev_yy])
    else:
        hist = model.fit(train_x, train_y, batch_size=BATCH_SIZE, epochs=EPOCHS, shuffle=True,
                         validation_data=[dev_x, dev_y], callbacks=call_list)
    print('hist:', hist.history)
    # 设置检查点，保存最好的记录
    # model.fit(train_x, train_y,batch_size=16,epochs=process_data.EPOCHS, validation_split=0.2,callbacks =call_list)
    model.save(MODEL_PATH)
    print('saving model', MODEL_PATH)
    # evaluate
    # loss,score = model.evaluate(test_x,test_y,verbose=1)
    # print('acc:',score)

    # 是否需要转换
    if DATA_SET == 'ccks' or DATA_SET == 'ccks2':
        if IOBES == 'iobes':
            iobes = False
        else:
            iobes = True

        ccks_labels, ccks_pre_labels = val.pre_label(process_data.TEST_PATH, MODEL_PATH, PRE_PATH, model_type, DIC,
                                                     DATA_SET, iobe=iobes)
        p, r, f = ccks_eval.ccks2_eval(ccks_pre_labels, ccks_labels, DIC, DATA_SET)
        ccks_eval.result(p, r, f, SCORE_PATH, DATA_SET)


def evaluater(TEST_PATH_, PRE_PATH_, SCORE_PATH_):
    if DATA_SET == 'ccks' or DATA_SET == 'ccks2':
        if IOBES == 'iobes':
            iobes = False
        else:
            iobes = True
        print('model:', MODEL)
        print('model path:', MODEL_PATH)
        print('testdata:', TEST_PATH_)
        ccks_labels, ccks_pre_labels = val.pre_label(TEST_PATH_, MODEL_PATH, PRE_PATH_, MODEL, DIC, DATA_SET,
                                                     iobe=iobes)
        # print(ccks_labels)
        # print(ccks_pre_labels)
        p, r, f = ccks_eval.ccks2_eval(ccks_pre_labels, ccks_labels, DIC, DATA_SET)
        ccks_eval.result(p, r, f, SCORE_PATH_, DATA_SET)


def pre_(TEST_PATH_, PRE_PATH_, SCORE_PATH_):
    """
    预测标签（未标注数据）
    """

    if DATA_SET == 'ccks' or DATA_SET == 'ccks2':
        if IOBES == 'iobes':
            iobes = False
        else:
            iobes = True
        print('model:', MODEL)
        print('model path:', MODEL_PATH)
        ccks_labels, ccks_pre_labels = val.pre_labels(TEST_PATH_, MODEL_PATH, PRE_PATH_, MODEL, DIC, DATA_SET,
                                                      iobe=iobes)
        p, r, f = ccks_eval.ccks2_eval(ccks_pre_labels, ccks_labels, DIC, DATA_SET)
        ccks_eval.result(p, r, f, SCORE_PATH_)


if __name__ == "__main__":

    if MODE == 'train':
        print('*' * 20, 'start training ...', '*' * 20)
        begin = time.clock()
        starttime = datetime.datetime.now()
        train()
        endtime = datetime.datetime.now()
        seconds = (endtime - starttime).seconds
        timestr = str(seconds // 3600) + '小时' + str((seconds % 3600) / 60) + '分钟' + str(seconds % 60)
        print('总耗时长：', time.clock() - begin)
        print('运行时间：', timestr)

        if not os.path.exists(RESULT_PATH + str(endtime) + '/' + timestr):
            os.mkdir(RESULT_PATH + timestr)


    elif MODE == 'val':
        print('*' * 20, ' val ...', '*' * 20)
        starttime = datetime.datetime.now()
        evaluater(process_data.TEST_PATH, './pre.txt', './score.txt')
        endtime = datetime.datetime.now()
        seconds = (endtime - starttime).seconds
        timestr = str(seconds // 3600) + '小时' + str(seconds // 60) + '分钟' + str(seconds % 60)
        print(timestr)
        if not os.path.exists(RESULT_PATH + str(endtime) + '/' + timestr):
            os.mkdir(RESULT_PATH + timestr)
