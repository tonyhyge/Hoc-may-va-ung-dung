import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib
from flask import Flask, Response
import seaborn as sns
import io
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
matplotlib.use("SVG")
app = Flask(__name__)
# nhập thư viện
import numpy as np
import pandas as pd

# tạo hàm lấy dữ liệu
def loadCsv(filename) -> pd.DataFrame:
    '''Code here'''
    return pd.read_csv(filename)
# tạo hàm biến đổi cột định tính, dùng phương pháp one hot
def transform(data, columns_trans): # data dạng dataframe, data_trans là cột cần biến đổi --> dạng Series, nhiều cột cần biến đổi thì bỏ vào list
    for i in columns_trans:
        unique = data[i].unique() + '-' + i # trả lại mảng
        # tạo ma trận 0
        matrix_0 = np.zeros((len(data), len(unique)), dtype = int)
        frame_0 = pd.DataFrame(matrix_0, columns = unique)
        for index, value in enumerate(data[i]):
            frame_0.at[index, value + '-' + i] = 1
        data[unique] = frame_0
    return data # trả lại data truyền vào nhưng đã bị biến đổi
# tạo hàm scale dữ liệu về [0,1] (min max scaler)
def scale_data(data, columns_scale): # columns_scale là cột cần scale, nếu nhiều bỏ vào list ['a', 'b']
    for i in columns_scale:  
        _max = data[i].max()
        _min = data[i].min()
        '''Code here'''
        val= data[i]
        data[i] = (val - _min) / (_max - _min)
    return data # --> trả về frame
# hàm tính khoảng cách Cosine 
def cosine_distance(train_X, test_X): # cả 2 đều dạng mảng
    dict_distance = dict()
    for index, value in enumerate(test_X, start = 1):
        for j in train_X:
            result = np.sqrt(np.sum((j - value)**2))
            if index not in dict_distance:
                dict_distance[index] = [result]
            else:
                dict_distance[index].append(result)
    return dict_distance # {1: [6.0, 5.0], 2: [4.6, 3.1]}
# hàm gán kết quả theo k
def pred_test(k, train_X, test_X, train_y): # train_X, test_X là mảng, train_y là Series
    lst_predict = list()
    dict_distance = cosine_distance(train_X, test_X)
    train_y = train_y.to_frame(name = 'target').reset_index(drop = True) # train_y là frame
    frame_concat = pd.concat([pd.DataFrame(dict_distance), train_y], axis = 1)
    for i in range(1, len(dict_distance) + 1):
        sort_distance = frame_concat[[i, 'target']].sort_values(by = i, ascending = True)[:k] # sắp xếp và lấy k
        target_predict = sort_distance['target'].value_counts(ascending = False).index[0]
        lst_predict.append([i, target_predict])
    return lst_predict

@app.route('/print-plot')
def plot_png():
    data = loadCsv('/Users/phamquangminh/Documents/tài liệu/university/HK1-Y3/Học máy và ứng dụng/lab3/drug200.csv')
    df = transform(data, ['Sex', 'BP', 'Cholesterol']).drop(['Sex', 'BP', 'Cholesterol'], axis = 1)

    scale_data(df, ['Age', 'Na_to_K'])
    data_X = df.drop(['Drug'], axis = 1).values
    data_y = df['Drug']
    X_train, X_test, y_train, y_test = train_test_split(data_X, data_y, test_size = 0.2, random_state = 0)
    test_pred = pred_test(6, X_train, X_test, y_train)
    df_test_pred = pd.DataFrame(test_pred).drop([0], axis = 1)
    df_test_pred.index = range(1, len(test_pred) + 1)
    df_test_pred.columns = ['Predict']
    df_actual = pd.DataFrame(y_test)
    df_actual.index = range(1, len(y_test) + 1)
    df_actual.columns = ['Actual']

    conf_matrix = confusion_matrix(df_actual, df_test_pred, labels=['drugA', 'drugB', 'drugC', 'drugX', 'DrugY'])
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=['drugA', 'drugB', 'drugC', 'drugX', 'DrugY'])


    fig, ax = plt.subplots(figsize=(10, 7))
    disp.plot(cmap=plt.cm.summer, ax=ax)



    output = io.BytesIO()
    plt.savefig(output, format='png')
    output.seek(0)

    return Response(output.getvalue(), mimetype='image/png')


if __name__ == '__main__':
    app.run(debug=True)