from sklearn.metrics import f1_score, roc_auc_score, accuracy_score

def compute_accuracy(model, data, labels):
    pred = (model.predict(data) > 0.5).astype("int32")
    score = accuracy_score(labels, pred)
    return score

def compute_area_under_curve(model, data, labels):
    pred = (model.predict(data) > 0.5).astype("int32")
    score = roc_auc_score(labels, pred)
    return score

def compute_f1(model, data, labels):
    pred = (model.predict(data) > 0.5).astype("int32")
    score = f1_score(labels, pred)
    return score

def compute_all_scores(model, data, labels):
    acc = compute_accuracy(model, data, labels)
    auc = compute_area_under_curve(model, data, labels)
    f1 = compute_f1(model, data, labels)
    s = "{} - acc: {:.2f}; auc: {:.2f}; f1: {:.2f}".format(model.name, acc, auc, f1)
    print(s)
    return acc, auc, f1

