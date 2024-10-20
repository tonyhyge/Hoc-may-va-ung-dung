
# import import_ipynb

# import decisionTree
# import randomForest





from sklearn.preprocessing import LabelEncoder
from flask import Flask, render_template, Response
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from io import BytesIO
import base64
import matplotlib
def split_node(column, threshold_split):  
    left_node = column[column <= threshold_split].index  
    right_node = column[column > threshold_split].index  
    return left_node, right_node 
def entropy(y_target):  
    values, counts = np.unique(y_target, return_counts = True) 
    result = -np.sum([(count / len(y_target)) * np.log2(count / len(y_target)) for count in counts])
    return result  


def most_value(y_target):  
    value = y_target.value_counts().idxmax()  
    return value 

def best_split(dataX, target, feature_id):  
    best_ig = -1  
    best_feature = None 
    best_threshold = None
    for _id in feature_id:
        column= dataX.iloc[:, _id] 
        thresholds = set(column)
        for threshold in thresholds:  
            ig = info_gain(column, target, threshold) 
            if ig > best_ig: 
                best_ig = ig 
                best_feature = dataX.columns[_id] 
                best_threshold = threshold 
    return best_feature, best_threshold  

def info_gain(column, target, threshold_split):  
        entropy_start = entropy(target)  

        left_node, right_node = split_node(column, threshold_split)  

        n_target = len(target)  
        n_left = len(left_node)  
        n_right = len(right_node)  

        
        entropy_left = entropy(target[left_node])  
        entropy_right = entropy(target[right_node]) 

        
        weight_entropy = ((n_left / n_target) * entropy_left) + ((n_right / n_target) * entropy_right)

        
        ig = entropy_start - weight_entropy
        return ig

class DecisionTreeClass:
    def __init__(self, min_samples_split = 2, max_depth = 10, n_features = None):
        self.min_samples_split = min_samples_split  
        self.max_depth = max_depth  
        self.root =  None 
        self.n_features = n_features 
    def grow_tree(self, X, y, depth = 0):  
        n_samples, n_feats = X.shape   


        n_classes = len(np.unique(y))

        

        
        
        if n_classes == 1 or n_samples < self.min_samples_split or depth >= self.max_depth:
            leaf_value = most_value(y)
            return Node(value = leaf_value) 
            
        
        feature_id = np.random.choice(n_feats, self.n_features, replace = False)
        
        
        best_feature, best_threshold = best_split(X, y, feature_id)
        

        
        
        left_node = X[best_feature] <= best_threshold  
        right_node = X[best_feature] > best_threshold 

        
        
        
        left = self.grow_tree(X.loc[left_node], y.loc[left_node], depth + 1) 
        right = self.grow_tree(X.loc[right_node], y.loc[right_node], depth + 1) 

        
        return Node(best_feature, best_threshold, left, right)

    def fit(self, X, y):  
        
        
        y = y.astype(str)

        self.n_features = X.shape[1] if self.n_features is None else min(X.shape[1], self.n_features)
        self.root = self.grow_tree(X, y)  

    def traverse_tree(self, x, node):  
        
        if node.is_leaf_node():

            return node.value

        if x[node.feature] <= node.threshold:
            return self.traverse_tree(x, node.left)
        return self.traverse_tree(x, node.right)
    
    def predict(self, X):  
        return np.array([self.traverse_tree(x, self.root) for index, x in X.iterrows()]) 

class Node:
    def __init__(self, feature = None, threshold = None, left = None, right = None, *, value = None): 
        self.feature =  feature 
        self.threshold =   threshold
        self.left =   left
        self.right =   right
        self.value =   value

    def is_leaf_node(self): 
        return self.value is not None  
    

def print_tree(node, indent = ""):
    
    if node.is_leaf_node():
        print(f"{indent}Leaf: {node.value}")
        return
    
    
    print(f"{indent}Node: If {node.feature} <= {node.threshold:.2f}")

    
    print(f"{indent}  True:")
    print_tree(node.left, indent + "    ")

    
    print(f"{indent}  False:")
    print_tree(node.right, indent + "    ")
    
def accuracy(y_actual, y_pred): 
    acc = np.sum(y_pred.astype(float)==y_actual)/len(y_actual)
    return acc*100

def bootstrap(X, y): 
    n_sample = X.shape[0]
    _id = np.random.choice(n_sample, n_sample, replace = True) 
    return X.iloc[_id], y.iloc[_id] 
    

class RandomForest:
    def __init__(self, n_trees = 5, max_depth = 10, min_samples_split = 2, n_features = None):
        self.n_trees = n_trees 
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features
        self.trees = []

    def fit(self, X, y):  
        self.trees = [] 
        for i in range(self.n_trees):
            
            tree = DecisionTreeClass(min_samples_split = self.min_samples_split, max_depth = self.max_depth, n_features = self.n_features)
            X_sample, y_sample = bootstrap(X, y) 
            tree.fit(X_sample, y_sample) 
            self.trees.append(tree) 

    def predict(self, X):  
        
        arr_pred = np.array([tree.predict(X) for tree in self.trees])
        final_pred = []
        for i in range(arr_pred.shape[1]): 
            sample_pred = arr_pred[:, i] 
            final_pred.append(most_value(pd.Series(sample_pred))) 
        return np.array(final_pred)  




# Custom Decision Tree and Random Forest Class (as defined in the uploaded file)





matplotlib.use("SVG")

app = Flask(__name__)

# Route to display the classification report and plot
@app.route('/')
def index():
    # Load and preprocess the data
    data = pd.read_csv("/Users/phamquangminh/Documents/tài liệu/university/HK1-Y3/Học máy và ứng dụng/week4/drug200.csv")
  # Replace with correct file path
    X = data.iloc[:, :-1]
    y = data['Drug']

    # Preprocessing as per the uploaded code
    X["Sex"] = X["Sex"].replace({'M': 0, 'F': 1})
    X["BP"] = X["BP"].replace({'HIGH': 2, 'NORMAL': 1, 'LOW': 0})
    X["Cholesterol"] = X["Cholesterol"].replace({'HIGH': 1, 'NORMAL': 0})
    y = y.replace({'drugA': 0, 'drugB': 1, 'drugC': 2, 'drugX': 3, 'DrugY': 4})

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Train Decision Tree and Random Forest models
    decisionTree = DecisionTreeClass(min_samples_split=2, max_depth=10)
    decisionTree.fit(X_train, y_train)
    y_pred_dc = decisionTree.predict(X_test)

    randomForest = RandomForest(n_trees=3, n_features=4)
    randomForest.fit(X_train, y_train)
    y_pred_rf = randomForest.predict(X_test)

    # Convert predictions and test set to string for classification report
    y_test_str = y_test.astype(str)
    y_pred_dc_str = y_pred_dc.astype(str)
    y_pred_rf_str = y_pred_rf.astype(str)

    # Generate classification reports
    report_dc = classification_report(y_test_str, y_pred_dc_str, output_dict=True)
    report_rf = classification_report(y_test_str, y_pred_rf_str, output_dict=True)

    # Plot accuracy of both models using Matplotlib
    labels = ['Decision Tree', 'Random Forest']
    accuracy = [report_dc['accuracy'], report_rf['accuracy']]

    plt.bar(labels, accuracy, color=['blue', 'green'])
    plt.ylabel('Accuracy')
    plt.title(' Decision Tree vs Random Forest')

    # Save plot to a BytesIO object
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

    # Return the plot and the classification report as part of HTML
    return render_template('index.html', img_data=img_base64,
                           report_dc=classification_report(y_test_str, y_pred_dc_str),
                           report_rf=classification_report(y_test_str, y_pred_rf_str))


if __name__ == "__main__":
    app.run(debug=True,port=5002)