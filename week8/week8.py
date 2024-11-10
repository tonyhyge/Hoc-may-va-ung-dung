
from flask import Flask, Response
from sklearn.model_selection import train_test_split

import matplotlib
matplotlib.use("SVG")

app = Flask(__name__)
from sklearn.datasets import load_wine 
import numpy as np
import pandas as pd
@app.route("/")
def plot_png():
    def euclidean_distance(a, b):
        return np.sqrt(np.sum(np.power(a-b,2) ))#code here
    def knn_predict(X_train, y_train, X_test, k=5):
        y_pred = []
        for test_point in X_test:
            distances = [euclidean_distance(test_point, x) for x in X_train]
            k_indices = np.argsort(distances)[:k]
            k_nearest_labels = [y_train[i] for i in k_indices]

            most_common = max(set(k_nearest_labels), key=k_nearest_labels.count)
            y_pred.append(most_common)
            #code here
        return np.array(y_pred)
    # Dự đoán trên tập kiểm tra với k = 5
    # Định nghĩa hàm confusion_matrix
    def confusion_matrix(y_true, y_pred):
        TP = np.sum((y_true == True) & (y_pred == True))
        TN = np.sum((y_true == False) & (y_pred == False))
        FP = np.sum((y_true == False) & (y_pred == True))
        FN = np.sum((y_true == True) & (y_pred == False))
        print(np.sum((y_true == True)),np.sum(y_pred == True))
        return [TP, FP, FN, TN]

    # Hàm tính toán và in các chỉ số
    def evaluate_model(y_test, y_pred):
        cm = confusion_matrix(y_test, y_pred)

        TN, FP, FN, TP = cm
        accuracy = (TP+TN)/(TP+TN+FP+FN)
        recall = TP/(TP+FN)#code here
        specificity = TN/(TN+FP) #code here 
        precision = TP/(TP+FP)#code here
        f1 = (2*precision*recall)/(precision+recall) 
        print(f"Confusion Matrix:\n{cm}")
        print(f"Accuracy: {accuracy:.2f}")
        print(f"Recall: {recall:.2f}")
        print(f"Specificity: {specificity:.2f}")
        print(f"Precision: {precision:.2f}")
        print(f"F1 Score: {f1:.2f}")
        lis = [f"Confusion Matrix:  {cm}", f"Accuracy: {accuracy:.2f}", f"Recall: {recall:.2f}", f"Specificity: {specificity:.2f}", f"Precision: {precision:.2f}", f"F1 Score: {f1:.2f}"]

        return lis
    # Đánh giá mô hình KNN

    #code here
    wine =load_wine()
    X,y= wine.data,wine.target
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)#code here
    y_pred_knn = knn_predict(X_train, y_train, X_test, k=5)




    return evaluate_model(y_test,y_pred_knn)

if __name__ == '__main__':
    app.run(debug=True, port=5003)
