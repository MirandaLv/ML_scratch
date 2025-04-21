
# calculate the AUC and ROC from scratch

# true positive rate: TP / TP + FN (recall)
# false positive rate: FP / FP + TN (1 - precision)

def calculate_auc_roc(y_true, y_score):

    thresholds = sorted(set(y_score), reverse=True) # decending order
    thresholds.append(min(y_scores) - 1)  # Ensure final threshold catches all

    tpr_list = []
    fpr_list = []

    p = sum(y_true) # count of positive sample 1
    n = len(y_true) - p # count of negative sample 0

    for threshold in thresholds: # using the threshold to assign 1/0 to y_score (probability)
        tp = fp = tn = fn = 0
        for i in range(len(y_score)):
            pred = 1 if y_score[i] >= threshold else 0
            actual = y_true[i]
            if pred == 1 and actual == 1:
                tp += 1
            elif pred == 1 and actual == 0:
                fp += 1
            elif pred == 0 and actual == 0:
                tn += 1
            elif pred == 0 and actual == 1:
                fn += 1

        tpr = tp / p if p!=0 else 0
        fpr = fp / n if n!=0 else 0

        tpr_list.append(tpr)
        fpr_list.append(fpr)

    auc = 0.0 # area under the curve
    for i in range(1, len(fpr_list)):
        x_diff = fpr_list[i] - fpr_list[i-1] # h
        y_diff = (tpr_list[i] + tpr_list[i-1]) / 2
        auc += x_diff * y_diff

    return fpr_list, tpr_list, auc

y_true = [0, 0, 1, 1]
y_scores = [0.1, 0.4, 0.35, 0.8]

fpr, tpr, auc = calculate_auc_roc(y_true, y_scores)

print("FPR:", fpr)
print("TPR:", tpr)
print("AUC:", auc)



