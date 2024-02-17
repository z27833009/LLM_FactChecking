from sklearn.metrics import precision_score, recall_score, f1_score
def evaluation(labels ,predictions):
        
    precision = precision_score(labels,
                                predictions,
                                average='weighted')
    recall = recall_score(labels,
                        predictions,
                        average='weighted')
    f1 = f1_score(labels, predictions, average='weighted')

    # print(labels.tolist())
    # print(predictions)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1:", f1)