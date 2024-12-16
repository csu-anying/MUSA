# coding:utf-8
import bilsm_crf_model
import process_data
# import ccks_eval
import numpy as np
import pickle


# DIC = process_data.DICC
def pre_label(test_data_path, model_path, data_path, model_type, DIC, DATA_SET, iobe=True):
    """
    转换为三列IOBES数据
    :param test_data_path:
    :param model_path:
    :return:
    """
    data = process_data._parse_data(open(test_data_path, 'rb'))
    if DATA_SET == 'ccks':
        # 切分为句子
        print('ccks')
        data = process_data.parse_data_sentence(data)
        maxl = 200
    else:
        print('ccks2')
        (train_x, train_y), (test_x, test_y), dev_x = process_data.load_data()
        maxl = max(max(len(s) for s in train_x), max(len(s) for s in test_x))

    if DIC:
        if model_type == 'base_':
            # print('*' * 20, 'bilstm_crf_base_', '*' * 20)
            model, (vocab, vocab0, vocab1, chunk_tags) = bilsm_crf_model.create_model_base(train=False)

        elif model_type == 'self_':
            # print('*' * 20, 'bilstm_crf_self_', '*' * 20)
            model, (vocab, vocab0, vocab1, chunk_tags) = bilsm_crf_model.create_model_att(train=False)

        elif model_type == 'base':
            model, (vocab, vocab0, vocab1, chunk_tags) = bilsm_crf_model.create_model_base_(maxlen=maxl, train=False)

        elif model_type == 'self':
            model, (vocab, vocab0, vocab1, chunk_tags) = bilsm_crf_model.create_model_self(maxlen=maxl, train=False)
        elif model_type == 'mul_self':
            model, (vocab, vocab0, vocab1, chunk_tags) = bilsm_crf_model.create_model_mul_self(maxlen=maxl, train=False)

        vocab = vocab1
    else:
        if model_type == 'base_':
            print('*' * 20, 'bilstm_crf_base_', '*' * 20)
            model, (vocab, chunk_tags) = bilsm_crf_model.create_model_base(train=False)

        elif model_type == 'self_':
            print('*' * 20, 'bilstm_crf_self_', '*' * 20)
            model, (vocab, chunk_tags) = bilsm_crf_model.create_model_att(train=False)

        elif model_type == 'base':
            model, (vocab, chunk_tags) = bilsm_crf_model.create_model_base_(maxlen=maxl, train=False)
        elif model_type == 'self':
            print('maxl=', maxl)
            model, (vocab, chunk_tags) = bilsm_crf_model.create_model_self(maxlen=maxl, train=False)
        elif model_type == 'mul_self':
            model, (vocab, chunk_tags) = bilsm_crf_model.create_model_mul_self(maxlen=maxl, train=False)

    print(model_path)
    model.load_weights(model_path)
    predict_texts = []
    raw_pre_tag = []
    raw_result_tags = []
    ccks_labels = []
    ccks_pre_labels = []

    for test in data:
        label = []
        if DIC:
            predict_text = []
            for row in test:
                label.append(row[-1])
                # 与词典匹配
                predict_text.append(str(row[0]) + str(row[1]))
        else:
            predict_text = ''
            for row in test:
                label.append(row[1])
                predict_text += row[0]

        predict_texts.append(predict_text)

        strs, length = process_data.process_data(predict_text, vocab, maxl)
        # 返回的str是数字序列
        temp = model.predict(strs)
        raw = temp[0][-length:]
        result = [np.argmax(row) for row in raw]
        result_tags = [chunk_tags[i] for i in result]
        # print(strs)
        # print(raw_result_tags)
        raw_pre_tag.append(label)
        raw_result_tags.append(result_tags)
        # 标签为iob则转化为iobe
        if iobe:
            label = process_data.iob_iobes(label)
            result_tags = process_data.iob_iobes(result_tags)

        m1 = process_data.pre_ccks_label(predict_text, label)
        m2 = process_data.pre_ccks_label(predict_text, result_tags)

        ccks_labels.append(m1)
        ccks_pre_labels.append(m2)
        # print(m1)
    # 保留结果
    # print(raw_result_tags)
    write_pre(data_path, predict_texts, raw_pre_tag, raw_result_tags, DIC, iobe)
    print("结果已保存！")
    return ccks_labels, ccks_pre_labels


def write_pre(file_path, predict_texts, raw_pre_tag, raw_result_tags, DIC, iobe=True):
    f = open(file_path, 'w', encoding='utf-8')
    for text, rawp, rawr in zip(predict_texts, raw_pre_tag, raw_result_tags):
        if iobe:
            rawp = process_data.iob_iobes(rawp)
            rawr = process_data.iob_iobes(rawr)

        for texti, rawpi, rawri in zip(text, rawp, rawr):
            if DIC:
                # print(rawpi)
                f.write(texti + ' ' + rawpi + ' ' + rawri + '\n')
            else:
                f.write(texti + ' ' + rawpi + ' ' + rawri + '\n')
        f.write('\n')


def _pre_tag(iobes_data_path):
    """
    接收三列的数据
    IOBES标签数据
    """
    print('加载文件：', iobes_data_path)
    data = process_data._parse_data(open(iobes_data_path, 'rb'))
    predict_texts = []
    raw_pre_tag = []
    raw_result_tags = []
    ccks_labels = []
    ccks_pre_labels = []

    result_tags = []
    l = 2
    for test in data:
        label = []
        predict_text = ''
        result_tags = []
        for row in test:
            label.append(row[1])
            predict_text += row[0]
            result_tags.append(row[2])
        predict_texts.append(predict_text)

        raw_pre_tag.append(label)
        raw_result_tags.append(result_tags)

        m1 = process_data.pre_ccks_label(predict_text, label)
        m2 = process_data.pre_ccks_label(predict_text, result_tags)

        ccks_labels.append(m1)
        ccks_pre_labels.append(m2)
        # print(m1)
    return ccks_labels, ccks_pre_labels


"""
返回预测标签
"""


def pre_labels(test_data_path, model_path, data_path, model_type, DIC, DATA_SET, iobe=True):
    """
    未标注数据打标
    仅仅返回预测标签
    """
    data = process_data._parse_data(open(test_data_path, 'rb'))
    if DATA_SET == 'ccks':
        # 切分为句子

        data = process_data.parse_data_sentence(data)
        maxl = 200
    else:
        (train_x, train_y), (dev_x, dev_y), (test_x, test_y) = process_data.load_data()
        maxl = max(max(len(s) for s in train_x), max(len(s) for s in dev_x), max(len(s) for s in test_x))
        print(maxl)
    # data = process_data._parse_data(open(test_data_path, 'rb'))
    # data = process_data.parse_data_sentence(data)
    if DIC:
        if model_type == 'base':
            model, (vocab, chunk_tags) = bilsm_crf_model.create_model_base(train=False)
        elif model_type == 'base_':
            model, (vocab, vocab0, vocab1, chunk_tags) = bilsm_crf_model.create_model_base_(maxlen=maxl, train=False)
        elif model_type == 'self':
            model, (vocab, vocab0, vocab1, chunk_tags) = bilsm_crf_model.create_model_self(maxlen=maxl, train=False)
        elif model_type == 'mul_self':
            model, (vocab, vocab0, vocab1, chunk_tags) = bilsm_crf_model.create_model_mul_self(maxlen=maxl, train=False)
    else:
        if model_type == 'base':
            model, (vocab, chunk_tags) = bilsm_crf_model.create_model_base(train=False)
        elif model_type == 'base_':
            model, (vocab, chunk_tags) = bilsm_crf_model.create_model_base_(maxlen=maxl, train=False)
        elif model_type == 'self':
            model, (vocab, chunk_tags) = bilsm_crf_model.create_model_self(maxlen=maxl, train=False)
        elif model_type == 'mul_self':
            model, (vocab, chunk_tags) = bilsm_crf_model.create_model_mul_self(maxlen=maxl, train=False)

    model.load_weights(model_path)
    predict_texts = []
    raw_result_tags = []
    ccks_pre_labels = []

    for test in data:
        label = []

        if DIC:
            predict_text = []
            for row in test:
                predict_text.append((row[0], row[1]))

        else:
            predict_text = ''
            for row in test:
                predict_text += row[0]

        predict_texts.append(predict_text)

        strs, length = process_data.process_data(predict_text, vocab)
        raw = model.predict(strs)[0][-length:]
        result = [np.argmax(row) for row in raw]
        result_tags = [chunk_tags[i] for i in result]

        raw_result_tags.append(result_tags)
        # 标签为iob则转化为iobe
        if iobe:
            result_tags = process_data.iob_iobes(result_tags)

        m2 = process_data.pre_ccks_label(predict_text, result_tags)
        ccks_pre_labels.append(m2)
        # print(m1)
    # 保留结果
    write_pred(data_path, predict_texts, raw_result_tags, iobe)
    return ccks_pre_labels


def write_pred(file_path, predict_texts, raw_result_tags, DIC, iobe=True):
    """
    写入预测结果
    :param file_path:
    :param predict_texts:
    :param raw_result_tags:
    :param iobe:
    :return:
    """
    f = open(file_path, 'w', encoding='utf-8')
    for text, rawr in zip(predict_texts, raw_result_tags):
        if iobe:
            rawr = process_data.iob_iobes(rawr)

        for texti, rawri in zip(text, rawr):
            if DIC:
                f.write(texti[0] + ' ' + texti[1] + ' ' + rawri + '\n')
            else:
                f.write(texti + ' ' + rawri + '\n')
        f.write('\n')


if __name__ == "__main__":
    # """
    # test_path = 'data/test_set.txt'
    # model_path = 'model/crfmodel/crf_bio_ccksemb_remp.h5'
    # pre_path = 'result/pre/best.0.8891.pre.txt'
    # score_path = 'result/score.0.8891.txt'
    # model_type = 'base'

    # ccks_labels, ccks_pre_labels = pre_label(process_data.TEST_PATH, process_data.MODEL_PATH, process_data.PRE_PATH,
    #                                              process_data.MODEL, iobe=False)
    # # ccks_labels, ccks_pre_labels = pre_label(test_path,model_path,pre_path,
    # #                                          model_type, iobe=True)
    # p, r, f = ccks_eval.ccks2_eval(ccks_pre_labels, ccks_labels)
    # ccks_eval.result(p, r, f, process_data.SCORE_PATH)
    # ccks_eval.result(p, r, f, score_path)
    # """

    # pre_labels()
    """
    # 带评估的测试数据集
    ccks_labels,ccks_pre_labels = pre_label('data/test_set.txt',process_data.MODEL_PATH,process_data.PRE_PATH,True)
    
    p,r,f = ccks_eval.ccks2_eval(ccks_pre_labels,ccks_labels)
    print('p:',p)
    print('r:',r)
    print('f:',f)
    #
    """
    """
    # resampling
    0.8890999762526715
    p: {'disease_s': 0.8421052631578947, 'body_r': 0.9491978609625669, 'body_s': 0.8516042780748663, 'treatment_r': 0.8227848101265823, 'overall_s': 0.8801128349788434, 'treatment_s': 0.5949367088607594, 'symptom_r': 0.9079365079365079, 'exam_r': 0.9713804713804713, 'symptom_s': 0.8968253968253969, 'disease_r': 0.881578947368421, 'overall_r': 0.9360601786553832, 'exam_s': 0.9410774410774411}
    r: {'disease_s': 0.7619047619047619, 'body_r': 0.9354413702239789, 'body_s': 0.839262187088274, 'treatment_r': 0.9027777777777778, 'overall_s': 0.8982725527831094, 'treatment_s': 0.6527777777777778, 'symptom_r': 0.9896193771626297, 'exam_r': 0.9763113367174281, 'symptom_s': 0.9775086505190311, 'disease_r': 0.7976190476190477, 'overall_r': 0.9553742802303263, 'exam_s': 0.9458544839255499}
    f: {'disease_s': 0.8, 'body_r': 0.942269409422694, 'body_s': 0.8453881884538819, 'treatment_r': 0.8609271523178809, 'overall_s': 0.8890999762526715, 'treatment_s': 0.6225165562913907, 'symptom_r': 0.9470198675496689, 'exam_r': 0.9738396624472574, 'symptom_s': 0.935430463576159, 'disease_r': 0.8375, 'overall_r': 0.9456186179054856, 'exam_s': 0.9434599156118143}
    
    """
    """
    0.8748815165876778
    p: {'exam_r': 0.9605911330049262, 'symptom_s': 0.8929712460063898, 'treatment_s': 0.5287356321839081, 'overall_s': 0.8642322097378277, 'treatment_r': 0.8045977011494253, 'body_r': 0.9505347593582888, 'disease_s': 0.8181818181818182, 'body_s': 0.8342245989304813, 'overall_r': 0.9321161048689138, 'exam_s': 0.9244663382594417, 'symptom_r': 0.902555910543131, 'disease_r': 0.9090909090909091}
    r: {'exam_r': 0.9898477157360406, 'symptom_s': 0.967128027681661, 'treatment_s': 0.6388888888888888, 'overall_s': 0.8857965451055663, 'treatment_r': 0.9722222222222222, 'body_r': 0.9367588932806324, 'disease_s': 0.6428571428571429, 'body_s': 0.8221343873517787, 'overall_r': 0.9553742802303263, 'exam_s': 0.9526226734348562, 'symptom_r': 0.9775086505190311, 'disease_r': 0.7142857142857143}
    f: {'exam_r': 0.975, 'symptom_s': 0.9285714285714285, 'treatment_s': 0.5786163522012578, 'overall_s': 0.8748815165876778, 'treatment_r': 0.8805031446540881, 'body_r': 0.9435965494359655, 'overall_r': 0.9436018957345971, 'body_s': 0.8281353682813537, 'disease_s': 0.7200000000000001, 'exam_s': 0.9383333333333334, 'symptom_r': 0.9385382059800665, 'disease_r': 0.8}
    """
    """
    ccks-emb
    0.7950594693504117
    p: {'overall_r': 0.8801989150090416, 'treatment_r': 0.8064516129032258, 'body_s': 0.759235668789809, 'symptom_s': 0.8133971291866029, 'exam_r': 0.9058084772370487, 'disease_s': 0.8, 'disease_r': 0.9285714285714286, 'overall_s': 0.7857142857142857, 'treatment_s': 0.5161290322580645, 'symptom_r': 0.8548644338118022, 'body_r': 0.8840764331210191, 'exam_s': 0.8288854003139717}
    r: {'overall_r': 0.9013888888888889, 'treatment_r': 0.8928571428571429, 'body_s': 0.7525252525252525, 'symptom_s': 0.8838821490467937, 'exam_r': 0.9232, 'disease_s': 0.6829268292682927, 'disease_r': 0.7926829268292683, 'overall_s': 0.8046296296296296, 'treatment_s': 0.5714285714285714, 'symptom_r': 0.92894280762565, 'body_r': 0.8762626262626263, 'exam_s': 0.8448}
    f: {'treatment_s': 0.5423728813559322, 'treatment_r': 0.8474576271186439, 'body_s': 0.7558655675332911, 'symptom_s': 0.8471760797342192, 'exam_r': 0.9144215530903328, 'disease_s': 0.736842105263158, 'disease_r': 0.855263157894737, 'overall_s': 0.7950594693504117, 'overall_r': 0.8906678865507777, 'symptom_r': 0.8903654485049833, 'body_r': 0.8801521876981611, 'exam_s': 0.8367670364500793}
    """
    """
    0.7524339360222532
    p: {'overall_s': 0.7534818941504178, 'symptom_s': 0.7951612903225806, 'overall_r': 0.8676880222841226, 'disease_r': 0.9387755102040817, 'symptom_r': 0.8338709677419355, 'treatment_r': 0.8470588235294118, 'body_r': 0.8781793842034806, 'exam_s': 0.7856049004594181, 'body_s': 0.7322623828647925, 'exam_r': 0.885145482388974, 'disease_s': 0.5918367346938775, 'treatment_s': 0.4823529411764706}
    r: {'overall_s': 0.7513888888888889, 'symptom_s': 0.854419410745234, 'overall_r': 0.8652777777777778, 'disease_r': 0.5609756097560976, 'symptom_r': 0.8960138648180243, 'treatment_r': 0.8571428571428571, 'body_r': 0.8282828282828283, 'exam_s': 0.8208, 'body_s': 0.6906565656565656, 'exam_r': 0.9248, 'disease_s': 0.35365853658536583, 'treatment_s': 0.4880952380952381}
    f: {'overall_s': 0.7524339360222532, 'symptom_s': 0.8237259816207184, 'overall_r': 0.8664812239221141, 'disease_s': 0.44274809160305345, 'symptom_r': 0.8638262322472848, 'treatment_r': 0.8520710059171598, 'body_r': 0.8525016244314491, 'exam_r': 0.9045383411580593, 'body_s': 0.7108512020792723, 'exam_s': 0.8028169014084506, 'disease_r': 0.7022900763358778, 'treatment_s': 0.48520710059171596}
    
    """
    """
    0.7860742098030233
    p: {'symptom_r': 0.8569131832797428, 'exam_r': 0.8808049535603715, 'treatment_s': 0.5, 'overall_s': 0.7778785131459656, 'disease_r': 0.881578947368421, 'disease_s': 0.6447368421052632, 'treatment_r': 0.7659574468085106, 'body_s': 0.7643229166666666, 'exam_s': 0.8111455108359134, 'symptom_s': 0.8183279742765274, 'body_r': 0.8828125, 'overall_r': 0.8699002719854941}
    r: {'symptom_r': 0.9237435008665511, 'exam_r': 0.9104, 'treatment_s': 0.5595238095238095, 'overall_s': 0.7944444444444444, 'disease_r': 0.8170731707317073, 'disease_s': 0.5975609756097561, 'treatment_r': 0.8571428571428571, 'body_s': 0.7411616161616161, 'exam_s': 0.8384, 'symptom_s': 0.8821490467937608, 'body_r': 0.8560606060606061, 'overall_r': 0.888425925925926}
    f: {'symptom_r': 0.8890742285237698, 'exam_r': 0.8953579858379228, 'treatment_r': 0.8089887640449439, 'overall_s': 0.7860742098030233, 'disease_r': 0.8481012658227848, 'disease_s': 0.620253164556962, 'treatment_s': 0.5280898876404494, 'body_s': 0.7525641025641026, 'exam_s': 0.8245476003147127, 'symptom_s': 0.8490408673894912, 'body_r': 0.8692307692307693, 'overall_r': 0.8790655061841502}
    
    """
    """
    0.30670294335889187
    p: {'exam_s': 0.31386861313868614, 'body_s': 0.4128289473684211, 'overall_s': 0.3697078115682767, 'treatment_r': 0.8627450980392157, 'disease_r': 0.5833333333333334, 'symptom_r': 0.5283842794759825, 'symptom_s': 0.38427947598253276, 'overall_r': 0.5736434108527132, 'treatment_s': 0.37254901960784315, 'body_r': 0.625, 'disease_s': 0.16666666666666666, 'exam_r': 0.5273722627737226}
    r: {'exam_s': 0.23789764868603042, 'body_s': 0.2935672514619883, 'overall_s': 0.26204564666103125, 'treatment_r': 0.5116279069767442, 'disease_r': 0.07954545454545454, 'symptom_r': 0.3941368078175896, 'symptom_s': 0.28664495114006516, 'overall_r': 0.4065934065934066, 'treatment_s': 0.22093023255813954, 'body_r': 0.4444444444444444, 'disease_s': 0.022727272727272728, 'exam_r': 0.3997233748271093}
    f: {'exam_s': 0.27065302911093625, 'overall_s': 0.30670294335889187, 'exam_r': 0.45476003147128247, 'treatment_r': 0.6423357664233577, 'disease_r': 0.13999999999999999, 'body_s': 0.34313055365686945, 'symptom_s': 0.3283582089552239, 'overall_r': 0.4758842443729904, 'treatment_s': 0.2773722627737227, 'body_r': 0.5194805194805195, 'disease_s': 0.04, 'symptom_r': 0.4514925373134329}
    
    """

    """
    0.4388472884048105
    p: {'disease_r': 0.3684210526315789, 'disease_s': 0.2631578947368421, 'symptom_r': 0.6117216117216118, 'exam_s': 0.44108761329305135, 'treatment_s': 0.5571428571428572, 'exam_r': 0.5876132930513596, 'symptom_s': 0.5311355311355311, 'body_s': 0.463448275862069, 'body_r': 0.6289655172413793, 'overall_s': 0.4737873591376776, 'treatment_r': 0.7142857142857143, 'overall_r': 0.6090151886330231}
    r: {'disease_r': 0.1590909090909091, 'disease_s': 0.11363636363636363, 'symptom_r': 0.5439739413680782, 'exam_s': 0.40387275242047027, 'treatment_s': 0.45348837209302323, 'exam_r': 0.5380359612724758, 'symptom_s': 0.4723127035830619, 'body_s': 0.3929824561403509, 'body_r': 0.5333333333333333, 'overall_s': 0.4087066779374472, 'treatment_r': 0.5813953488372093, 'overall_r': 0.5253592561284869}
    f: {'disease_r': 0.22222222222222218, 'symptom_r': 0.5758620689655173, 'treatment_s': 0.5, 'exam_s': 0.4216606498194946, 'disease_s': 0.15873015873015872, 'body_s': 0.4253164556962025, 'symptom_s': 0.4999999999999999, 'body_r': 0.5772151898734176, 'exam_r': 0.5617328519855596, 'treatment_r': 0.6410256410256411, 'overall_r': 0.564102564102564, 'overall_s': 0.4388472884048105}
    
    """
    """
    # 三列包含BIOES标签
    ccks_labels,ccks_pre_labels, = _pre_tag(process_data.PRE_PATH)
    p,r,f = ccks_eval.ccks2_eval(ccks_pre_labels,ccks_labels)
    ccks_eval.result(p,r,f,process_data.SCORE_PATH)
    """
    """
    # pytorch
    0.8836717428087986
    p: {'symptom_r': 0.9, 'body_s': 0.8564294631710362, 'disease_s': 0.8428571428571429, 'overall_r': 0.9356477561388654, 'disease_r': 0.9142857142857143, 'overall_s': 0.884419983065199, 'exam_r': 0.9528936742934051, 'treatment_r': 0.875, 'treatment_s': 0.7045454545454546, 'body_r': 0.9575530586766542, 'symptom_s': 0.8954545454545455, 'exam_s': 0.9300134589502019}
    r: {'symptom_r': 0.9674267100977199, 'body_s': 0.8023391812865497, 'disease_s': 0.6704545454545454, 'overall_r': 0.9340659340659341, 'disease_r': 0.7272727272727273, 'overall_s': 0.8829247675401521, 'exam_r': 0.979253112033195, 'treatment_r': 0.8953488372093024, 'treatment_s': 0.7209302325581395, 'body_r': 0.8970760233918129, 'symptom_s': 0.9625407166123778, 'exam_s': 0.9557399723374828}
    f: {'symptom_r': 0.9324960753532183, 'body_s': 0.8285024154589373, 'disease_s': 0.7468354430379747, 'overall_r': 0.9348561759729274, 'disease_r': 0.810126582278481, 'overall_s': 0.8836717428087986, 'exam_r': 0.9658935879945428, 'treatment_r': 0.8850574712643678, 'treatment_s': 0.7126436781609196, 'body_r': 0.9263285024154589, 'symptom_s': 0.9277864992150707, 'exam_s': 0.9427012278308322}
    
    """
    """
    0.4891609283346085
    p: {'exam_s': 0.4401840490797546, 'body_s': 0.4583901773533424, 'symptom_r': 0.6018348623853211, 'disease_r': 0.4, 'exam_r': 0.5674846625766872, 'symptom_s': 0.5357798165137615, 'treatment_s': 0.4810126582278481, 'overall_s': 0.4703285924472781, 'disease_s': 0.2, 'treatment_r': 0.7468354430379747, 'body_r': 0.6098226466575716, 'overall_r': 0.5963707699852869}
    r: {'exam_s': 0.5394736842105263, 'body_s': 0.5137614678899083, 'symptom_r': 0.6188679245283019, 'disease_r': 0.14457831325301204, 'exam_r': 0.6954887218045113, 'symptom_s': 0.5509433962264151, 'treatment_s': 0.4578313253012048, 'overall_s': 0.5095642933049946, 'disease_s': 0.07228915662650602, 'treatment_r': 0.7108433734939759, 'body_r': 0.6834862385321101, 'overall_r': 0.6461211477151966}
    f: {'exam_s': 0.4847972972972973, 'body_s': 0.48449891852919974, 'symptom_r': 0.610232558139535, 'disease_r': 0.21238938053097345, 'exam_r': 0.625, 'symptom_s': 0.5432558139534883, 'treatment_s': 0.4691358024691358, 'overall_s': 0.4891609283346085, 'disease_s': 0.10619469026548672, 'treatment_r': 0.728395061728395, 'body_r': 0.6445565969718817, 'overall_r': 0.620249936240755}
    
    """
