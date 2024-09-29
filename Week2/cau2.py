from flask import Flask, Response
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
import numpy as np
import io
import matplotlib.pyplot as plt 
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib
matplotlib.use("SVG")

app = Flask(__name__)

@app.route('/print-plot')
def plot_png():
    data = pd.read_csv('/Users/phamquangminh/Documents/tài liệu/university/HK1-Y3/Học máy và ứng dụng/Code/drug.csv')

    label_encoder_sex = LabelEncoder()
    data['Sex'] = label_encoder_sex.fit_transform(data['Sex'])

    label_encoder_BP = LabelEncoder()
    data['BP'] = label_encoder_BP.fit_transform(data['BP'])

    label_encoder_chol = LabelEncoder()
    data['Cholesterol'] = label_encoder_chol.fit_transform(data['Cholesterol'])

    X = data[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']]

    y = data['Drug']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = GaussianNB()

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred,output_dict=True)


    fig, axes = plt.subplots(1, 1, figsize=(18, 12))  


    labels = list(report.keys())[:-3]  
    metrics = ['precision', 'recall', 'f1-score', 'support']


    data_bernoulli = np.array([[report[label][metric] for metric in metrics] for label in labels])
    cax_bernoulli = axes.matshow(data_bernoulli, cmap='coolwarm')
    fig.colorbar(cax_bernoulli, ax=axes)
    axes.set_xticks(range(len(metrics)))
    axes.set_xticklabels(metrics)
    axes.set_yticks(range(len(labels)))
    axes.set_yticklabels(labels)
    axes.set_title('Bernoulli Naive Bayes')


    for (i, j), val in np.ndenumerate(data_bernoulli):
        axes.text(j, i, f'{val:.2f}', ha='center', va='center', color='white')

    plt.tight_layout()


    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True, port=5003)
