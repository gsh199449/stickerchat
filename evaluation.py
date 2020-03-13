from metrics import recall_2at1, recall_at_k_new, precision_at_k, MRR, MAP
import numpy as np


def evaluation(pred_scores, true_scores, samples=10):
    '''
    :param pred_scores:     list of scores predicted by model
    :param true_scores:     list of ground truth labels, 1 or 0
    :return:
    '''

    num_sample = int(len(pred_scores) / samples)  # 1 positive and 9 negative
    score_list = np.split(np.array(pred_scores), num_sample, axis=0)
    recall_2_1 = recall_2at1(np.array(true_scores), np.array(pred_scores))
    recall_at_1 = recall_at_k_new(np.array(true_scores), np.array(pred_scores), 1)
    recall_at_2 = recall_at_k_new(np.array(true_scores), np.array(pred_scores), 2)
    recall_at_5 = recall_at_k_new(np.array(true_scores), np.array(pred_scores), 5)
    _mrr = MRR(np.array(true_scores), np.array(pred_scores))
    _map = MAP(np.array(true_scores), np.array(pred_scores))
    precision_at_1 = precision_at_k(np.array(true_scores), np.array(pred_scores), k=1)
    return {
        'MAP': _map,
        'MRR': _mrr,
        'p@1': precision_at_1,
        'r2@1': recall_2_1,
        'r@1': recall_at_1,
        'r@2': recall_at_2,
        'r@5': recall_at_5,
    }
