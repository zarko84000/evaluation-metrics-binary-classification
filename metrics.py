import numpy as np

def accuracy(y_true, y_pred):
    correct_counter = 0 
    for yt, yp in zip(y_true, y_pred):
        if yt == yp :
            correct_counter += 1
    return correct_counter / len(y_true)
    
def true_positive(y_true, y_pred):
    true_positive = 0 
    for yt, yp in zip(y_true, y_pred):
        if yt == 1 and yp == 1 :
            true_positive += 1
    return true_positive
    
def true_negative(y_true, y_pred):
    true_negative = 0 
    for yt, yp in zip(y_true, y_pred):
        if yt == 0 and yp == 0:
            true_negative += 1
    return true_negative

def false_positive(y_true, y_pred):
    fp = 0 
    for yt, yp in zip(y_true, y_pred):
        if yt == 0 and yp == 1:
            fp += 1
    return fp
    
def false_negative(y_true, y_pred):
    fn = 0 
    for yt, yp in zip(y_true, y_pred):
        if yt == 1 and yp == 0:
            fn += 1
    return fn
    
def accuracy_v2(y_true, y_pred):
    tp = true_positive(y_true, y_pred)
    tn = true_negative(y_true, y_pred)
    fp = false_positive(y_true,y_pred)
    fn = false_negative(y_true, y_pred)
    
    score = (tp+tn)/(tp+tn+fp+fn)
    return score
    
def precision(y_true, y_pred):
    tp = true_positive(y_true, y_pred)
    fp = false_positive(y_true, y_pred)

    precision = tp/(tp + fp)
    return precision

def recall(y_true, y_pred):
    tp = true_positive(y_true, y_pred)
    fn = false_negative(y_true, y_pred)
    recall = tp/(tp+fn)
    return recall

def tpr(y_true, y_pred):
    return recall(y_true, y_pred)
  
 
def fpr(y_true, y_pred):
    fp = false_positive(y_true, y_pred)
    tn = true_negative(y_true, y_pred)

    return fp / (tn + fp)
 
def log_loss(y_true, y_proba):
    epsilon = 1e-15
    loss = []
    for yt, yp in zip(y_true, y_proba):
        
        yp = np.clip(yp, epsilon, 1 - epsilon)

        temp_loss = - 1.0 * (
            yt * np.log(yp)
             + (1 - yt) * np.log(1 -  yp)
             )

        loss.append(temp_loss)
    
    return np.mean(loss) 
    
    
