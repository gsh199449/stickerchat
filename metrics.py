import numpy


def recall_2at1(labels, scores):
    num_sample = int(len(scores) / 10)  # 1 positive and 9 negative
    score_list = numpy.split(numpy.array(scores), num_sample, axis=0)
    label_list = numpy.split(numpy.array(labels), num_sample, axis=0)

    cnt, tot = 0, 0
    for score, label in zip(score_list, label_list):
        tot += 1
        pos_score, neg_score = 0, 0
        find_neg = False
        find_pos = False
        for i in range(10):
            if label[i] == 1:
                pos_score = score[i]
                find_pos = True
            if label[i] == 0:
                neg_score = score[i]
                find_neg = True
            if find_pos and find_neg:
                break
        if pos_score > neg_score:
            cnt += 1

    return cnt * 1.0 / tot


def recall_at_k_new(labels, scores, k=1, doc_num=10):
    scores = scores.reshape(-1, doc_num) # [batch, doc_num]
    labels = labels.reshape(-1, doc_num) # # [batch, doc_num]
    sorted, indices = numpy.sort(scores, 1), numpy.argsort(-scores, 1)
    count_nonzero = 0
    recall = 0
    for i in range(indices.shape[0]):
        num_rel = numpy.sum(labels[i])
        if num_rel==0: continue
        rel = 0
        for j in range(k):
            if labels[i, indices[i, j]] == 1:
                rel += 1
        recall += float(rel) / float(num_rel)
        count_nonzero += 1
    return float(recall) / count_nonzero


def precision_at_k(labels, scores, k=1, doc_num=10):
    
    scores = scores.reshape(-1,doc_num) # [batch, doc_num]
    labels = labels.reshape(-1,doc_num) # [batch, doc_num]

    sorted, indices = numpy.sort(scores, 1), numpy.argsort(-scores, 1)
    count_nonzero = 0
    precision = 0
    for i in range(indices.shape[0]):
        num_rel = numpy.sum(labels[i])
        if num_rel==0: continue
        rel = 0
        for j in range(k):
            if labels[i, indices[i, j]] == 1:
                rel += 1
        precision += float(rel) / float(k)
        count_nonzero += 1
    return precision / count_nonzero


def MAP(target, logits, k=10):
    """
    Compute mean average precision.
    :param target: 2d array [batch_size x num_clicks_per_query] true
    :param logits: 2d array [batch_size x num_clicks_per_query] pred
    :return: mean average precision [a float value]
    """
    assert logits.shape == target.shape

    target = target.reshape(-1,k)
    logits = logits.reshape(-1,k)
    
    sorted, indices = numpy.sort(logits, 1)[::-1], numpy.argsort(-logits, 1)
    count_nonzero = 0
    map_sum = 0
    for i in range(indices.shape[0]):
        average_precision = 0
        num_rel = 0
        for j in range(indices.shape[1]):
            if target[i, indices[i, j]] == 1:
                num_rel += 1
                average_precision += float(num_rel) / (j + 1)
        if num_rel==0: continue
        average_precision = average_precision / num_rel
        map_sum += average_precision
        count_nonzero += 1
    return float(map_sum) / count_nonzero


def MRR(target, logits, k=10):
    """
    Compute mean reciprocal rank.
    :param target: 2d array [batch_size x rel_docs_per_query]
    :param logits: 2d array [batch_size x rel_docs_per_query]
    :return: mean reciprocal rank [a float value]
    """
    assert logits.shape == target.shape
    target = target.reshape(-1,k)
    logits = logits.reshape(-1,k)

    sorted, indices = numpy.sort(logits, 1)[::-1], numpy.argsort(-logits, 1)
    count_nonzero=0
    reciprocal_rank = 0
    for i in range(indices.shape[0]):
        flag=0
        for j in range(indices.shape[1]):
            if target[i, indices[i, j]] == 1:
                reciprocal_rank += float(1.0) / (j + 1)
                flag=1
                break
        if flag: count_nonzero += 1

    return float(reciprocal_rank) / count_nonzero
