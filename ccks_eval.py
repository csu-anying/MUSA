# coding: utf-8
# from model_train import DIC
# DIC = False
# DIC = True

def result(p, r, f1, file_path, DATA_SET):
    # 输出结果

    score_ = ['body_s', 'disease_s', 'treatment_s', 'symptom_s', 'exam_s', 'overall_s']
    score_2 = []
    if DATA_SET == 'ccks2':
        score_2 = ['AP_s', 'IP_s', 'OP_s', 'SD_s', 'DG_s', 'overall_s']

    mat = "{:10}\t{:10}\t{:10}\t{:10}"
    f = open(file_path, 'w', encoding='utf-8')
    print(mat.format('\t', 'precision', 'recall', 'f1'))
    f.write(mat.format('\t', 'precision', 'recall', 'f1'))
    f.write('\n')
    for i, sc in enumerate(score_):
        if DATA_SET == 'ccks2':
            sc2 = score_2[i]
        else:
            sc2 = sc
        f.write(mat.format(sc2, round(p[sc], 4), round(r[sc], 4), round(f1[sc], 4)))
        f.write('\n')
        print(mat.format(sc2, round(p[sc], 4), round(r[sc], 4), round(f1[sc], 4)))


def Result(s, m):
    print(s)
    return m


def ccks2_eval(submission, truth, DIC, DATA_SET):
    """"
        data格式
        ['查体 3 4 CHECK', 'T 6 6 CHECK']
    """
    dict_tru = {}
    dict_sub = {}
    i = 0
    print("tru:", len(truth))
    print("sub:", len(submission))

    for tru_line, sub_line in zip(truth, submission):
        # 读取实体及实体类型，位置
        # 从前向后排列

        dict_tru[i] = tru_line
        dict_sub[i] = sub_line
        # print(i,tru_line)
        # print(i,sub_line)
        i += 1

    symptom_dict, disease_dict, exam_dict, treatment_dict, body_dict = {}, {}, {}, {}, {}
    symptom_g, disease_g, exam_g, treatment_g, body_g, overall_g = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    if DATA_SET == 'ccks':
        tags = ['SIGNS', 'DISEASE', 'CHECK', 'TREATMENT', 'BODY']
    elif DATA_SET == 'ccks2':
        tags = ['SD', 'IP', 'DG', 'OP', 'AP']
    for row_id in dict_tru:
        if row_id not in dict_sub:
            print("missing id: " + row_id)
            return Result(-1, "missing id: " + row_id)
        else:
            # 实体分隔符
            t_lst = dict_tru[row_id]

            for item in t_lst:
                # 在分割 文本 位置b e 类型
                item = item.split(' ')
                overall_g += 1
                if item[1] == '肌腱' or item[1] == '肌腱反射' or item[1] == '红素':
                    item[0] == item[0] + item.pop(1)
                if DIC:
                    item[3] = item[-1]
                if item[3] == tags[0]:
                    symptom_g += 1
                    if row_id not in symptom_dict:
                        symptom_dict[row_id] = []
                    symptom_dict[row_id].append(item[:-1])
                elif item[3] == tags[1]:
                    disease_g += 1
                    if row_id not in disease_dict:
                        disease_dict[row_id] = []
                    disease_dict[row_id].append(item[:-1])
                elif item[3] == tags[2]:
                    exam_g += 1
                    if row_id not in exam_dict:
                        exam_dict[row_id] = []
                    exam_dict[row_id].append(item[:-1])
                elif item[3] == tags[3]:
                    treatment_g += 1
                    if row_id not in treatment_dict:
                        treatment_dict[row_id] = []
                    treatment_dict[row_id].append(item[:-1])
                elif item[3] == tags[4]:
                    body_g += 1
                    if row_id not in body_dict:
                        body_dict[row_id] = []
                    body_dict[row_id].append(item[:-1])
                else:
                    print("row_id:" + str(row_id))
                    print("unknown label: " + str(item))
                    return Result(-1, "unknown label: " + str(item))

    symptom_s, disease_s, exam_s, treatment_s, body_s, overall_s = 0, 0, 0, 0, 0, 0
    symptom_r, disease_r, exam_r, treatment_r, body_r, overall_r = 0, 0, 0, 0, 0, 0
    predict, predict_symptom, predict_disease, predict_exam, predict_treatment, predict_body = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    for row_id in dict_sub:
        if row_id not in dict_tru:
            print("unknown id:" + row_id)
            return Result(-1, "unknown id:" + row_id)

        s_lst = set(dict_sub[row_id])
        predict += len(s_lst)

        for item in s_lst:
            if len(item) == 0:
                continue
            item = item.split(' ')

            # if len(item) >= 5:
            #     item[0] = item[0] + ' ' + item[1]
            #     item[1] = item[2]
            #     item[2] = item[3]
            #     item[3] = item[4]
            if len(item) < 4:
                for _item in item:
                    print(_item)
                # print(item)
                return Result(-1, "incorrect format around id: " + str(row_id))

            # 开始解析正确的文本
            if DIC:
                item[3] = item[-1]

            if item[3] == tags[4]:
                predict_body += 1
                if row_id not in body_dict:
                    continue
                if item[:-1] in body_dict[row_id]:
                    body_s += 1
                    overall_s += 1
                    body_r += 1
                    overall_r += 1
                else:
                    for gold in body_dict[row_id]:
                        if max(item[-3], gold[-3]) <= min(item[-2], gold[-2]):
                            body_r += 1
                            overall_r += 1
                            break
            elif item[3] == tags[0]:
                predict_symptom += 1
                if row_id not in symptom_dict:
                    continue
                if item[:-1] in symptom_dict[row_id]:
                    symptom_s += 1
                    overall_s += 1
                    symptom_r += 1
                    overall_r += 1
                else:
                    for gold in symptom_dict[row_id]:
                        if max(item[-3], gold[-3]) <= min(item[-2], gold[-2]):
                            symptom_r += 1
                            overall_r += 1
                            break
            elif item[3] == tags[1]:
                predict_disease += 1
                if row_id not in disease_dict:
                    continue
                if item[:-1] in disease_dict[row_id]:
                    disease_s += 1
                    overall_s += 1
                    disease_r += 1
                    overall_r += 1
                else:
                    for gold in disease_dict[row_id]:
                        if max(item[-3], gold[-3]) <= min(item[-2], gold[-2]):
                            disease_r += 1
                            overall_r += 1
                            break
            elif item[3] == tags[2]:
                predict_exam += 1
                if row_id not in exam_dict:
                    continue
                if item[:-1] in exam_dict[row_id]:
                    exam_s += 1
                    overall_s += 1
                    exam_r += 1
                    overall_r += 1
                else:
                    for gold in exam_dict[row_id]:
                        if max(item[-3], gold[-3]) <= min(item[-2], gold[-2]):
                            exam_r += 1
                            overall_r += 1
                            break
            elif item[3] == tags[3]:
                predict_treatment += 1
                if row_id not in treatment_dict:
                    continue
                if item[:-1] in treatment_dict[row_id]:
                    treatment_s += 1
                    overall_s += 1
                    treatment_r += 1
                    overall_r += 1
                else:
                    for gold in treatment_dict[row_id]:
                        if max(item[-3], gold[-3]) <= min(item[-2], gold[-2]):
                            treatment_r += 1
                            overall_r += 1
                            break

    precision, recall, f1 = {}, {}, {}

    precision['symptom_s'] = symptom_s / predict_symptom
    precision['disease_s'] = disease_s / predict_disease
    precision['exam_s'] = exam_s / predict_exam
    precision['treatment_s'] = treatment_s / predict_treatment
    precision['body_s'] = body_s / predict_body
    precision['overall_s'] = overall_s / predict

    precision['symptom_r'] = symptom_r / predict_symptom
    precision['disease_r'] = disease_r / predict_disease
    precision['exam_r'] = exam_r / predict_exam
    precision['treatment_r'] = treatment_r / predict_treatment
    precision['body_r'] = body_r / predict_body
    precision['overall_r'] = overall_r / predict

    recall['symptom_s'] = symptom_s / symptom_g
    recall['disease_s'] = disease_s / disease_g
    recall['exam_s'] = exam_s / exam_g
    recall['treatment_s'] = treatment_s / treatment_g
    recall['body_s'] = body_s / body_g
    recall['overall_s'] = overall_s / overall_g

    recall['symptom_r'] = symptom_r / symptom_g
    recall['disease_r'] = disease_r / disease_g
    recall['exam_r'] = exam_r / exam_g
    recall['treatment_r'] = treatment_r / treatment_g
    recall['body_r'] = body_r / body_g
    recall['overall_r'] = overall_r / overall_g

    # print('precision:',str(precision))
    # print('recall:',str(recall))
    # f值
    for item in precision:
        f1[item] = 2 * precision[item] * recall[item] / (precision[item] + recall[item]) \
            if (precision[item] + recall[item]) != 0 else 0

    return precision, recall, f1


if __name__ == "__main__":
    pass
