# Nhập thư viện
import numpy as np
import pandas as pd
from flask import Flask, Response
import io
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay

matplotlib.use("SVG")
app = Flask(__name__)
# Load dữ liệu
def loadExcel(filename) -> pd.DataFrame:
    '''Code here'''
    return pd.read_excel(filename)
# tạo tập train test (chia data_train (gộp X_train và y_train) và X_test và y_test)
def splitTrainTest(data, target, ratio = 0.25): # data --> frame
    from sklearn.model_selection import train_test_split
    data_X = data.drop([target], axis = 1)
    data_y = data[[target]]
    '''Code here'''
    X_train, X_test,y_train, y_test = train_test_split(data_X,data_y,test_size=ratio)
    data_train = pd.concat([X_train,y_train], axis = 1)
    return data_train, X_test, y_test # đều là dạng frame
# hàm tính trung bình của từng lớp trong biến target
def mean_class(data_train, target): # tên cột target, data_train là dạng pandas
    df_group = data_train.groupby(by = target).mean() # tất cả các cột đều dạng số, --> frame # sắp xếp theo bảng chữ cái tăng dần(mặc định)
    return df_group # kết quả là dataframe
# hàm dự đoán dùng khoảng cách euclid
def target_pred(data_group, data_test): # data_test ở dạng mảng, data_group là đã đem tính trung bình các lớp(là df_group)
    dict_ = dict()
    for index, value in enumerate(data_group.values):
        result = np.sqrt(np.sum(((data_test - value)**2), axis = 1)) # khoảng cách euclid
        if index in dict_:
            dict_.append(result)
        else:
            dict_[index] = result # Lưu ý chỗ này không phải là [result] vì result là dạng mảng, nếu thêm vào vậy thì nó chỉ có một phần tử.
    # dict_ kết quả dạng {0: [2.0], 1: [1.0540925533894596]}
    '''Code here'''
    df = pd.DataFrame(dict_)
    df.rename({0: 'Iris-setosa', 1: 'Iris-versicolor',2:"Iris-virginica"}, axis=1, inplace=True)

    return df.idxmin(axis = 1) # hàm này tìm cột chưá giá trị nhỏ nhất
    
##### Có thể phát triển: cho thêm một tham số metric vào hàm, nếu là euclid thì dùng khoảng cách euclid, manhattan thì dùng khoảng cách manhattan.
@app.route('/print-plot')
def plot_png():
    data = loadExcel('/Users/phamquangminh/Documents/tài liệu/university/HK1-Y3/Học máy và ứng dụng/lab3/Iris.xls')
    data_train, X_test, y_test = splitTrainTest(data, 'iris', ratio = 0.3)
    df_group = mean_class(data_train, 'iris')
    df1 = pd.DataFrame(target_pred(df_group, X_test.values), columns = ['Predict'])
    y_test.index = range(0, len(y_test))
    y_test.columns = ['Actual']
    df2 = pd.DataFrame(y_test)
    conf_matrix = confusion_matrix(df1, df2, labels=['Iris-setosa',  'Iris-versicolor',"Iris-virginica"])
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=['Iris-setosa',  'Iris-versicolor',"Iris-virginica"])


    fig, ax = plt.subplots(figsize=(10, 7))
    disp.plot(cmap=plt.cm.summer, ax=ax)



    output = io.BytesIO()
    plt.savefig(output, format='png')
    output.seek(0)

    return Response(output.getvalue(), mimetype='image/png')
if __name__ == '__main__':
    app.run(debug=True)