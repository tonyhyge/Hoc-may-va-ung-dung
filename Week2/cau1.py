from flask import Flask, Response
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
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
    file_path = '/Users/phamquangminh/Documents/tài liệu/university/HK1-Y3/Học máy và ứng dụng/Code/Education.csv'
    data = pd.read_csv(file_path)

    X = data['Text']
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(data['Label'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    vectorizer = CountVectorizer(stop_words='english')
    X_train_bow = vectorizer.fit_transform(X_train)
    X_test_bow = vectorizer.transform(X_test)

    bernoulli_nb = BernoulliNB()
    multinomial_nb = MultinomialNB()

    bernoulli_nb.fit(X_train_bow, y_train)
    multinomial_nb.fit(X_train_bow, y_train)

    y_pred_bernoulli = bernoulli_nb.predict(X_test_bow)
    y_pred_multinomial = multinomial_nb.predict(X_test_bow)

    report_bernoulli = classification_report(y_test, y_pred_bernoulli, output_dict=True)
    report_multinomial = classification_report(y_test, y_pred_multinomial, output_dict=True)

    fig, axes = plt.subplots(1, 2, figsize=(18, 12))

    labels = list(report_bernoulli.keys())[:-3]
    metrics = ['precision', 'recall', 'f1-score', 'support']

    data_bernoulli = np.array([[report_bernoulli[label][metric] for metric in metrics] for label in labels])
    cax_bernoulli = axes[0].matshow(data_bernoulli, cmap='coolwarm')
    fig.colorbar(cax_bernoulli, ax=axes[0])
    axes[0].set_xticks(range(len(metrics)))
    axes[0].set_xticklabels(metrics)
    axes[0].set_yticks(range(len(labels)))
    axes[0].set_yticklabels(labels)
    axes[0].set_title('Bernoulli Naive Bayes')

    for (i, j), val in np.ndenumerate(data_bernoulli):
        axes[0].text(j, i, f'{val:.2f}', ha='center', va='center', color='white')

    data_multinomial = np.array([[report_multinomial[label][metric] for metric in metrics] for label in labels])
    cax_multinomial = axes[1].matshow(data_multinomial, cmap='coolwarm')
    fig.colorbar(cax_multinomial, ax=axes[1])
    axes[1].set_xticks(range(len(metrics)))
    axes[1].set_xticklabels(metrics)
    axes[1].set_yticks(range(len(labels)))
    axes[1].set_yticklabels(labels)
    axes[1].set_title('Multinomial Naive Bayes')

    for (i, j), val in np.ndenumerate(data_multinomial):
        axes[1].text(j, i, f'{val:.2f}', ha='center', va='center', color='white')

    plt.tight_layout()

    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True, port=5002)
