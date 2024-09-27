from flask import Flask, Response
import numpy as np
import io
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

app = Flask(__name__)

@app.route('/print-plot')
def plot_png():
    X = np.array([180, 162, 183, 174, 160, 163, 180, 165, 175, 170, 170, 169,
                  168, 175, 169, 171, 155, 158, 175, 165]).reshape(-1,1)
    y = np.array([86, 55, 86.5, 70, 62, 54, 60, 72, 93, 89, 60, 82, 59, 75, 
                  56, 89, 45, 60, 60, 72]).reshape((-1,1))
    
    X = np.insert(X, 0, 1, axis=1)
    
    theta = np.linalg.inv(X.T @ X) @ (X.T @ y)
    
    x1 = 150
    y1 = theta[0] + theta[1] * x1
    x2 = 190
    y2 = theta[0] + theta[1] * x2

    fig = Figure()
    axis = fig.add_subplot(1, 1, 1)

    axis.plot([x1, x2], [y1, y2], 'r-')
    axis.plot(X[:,1], y[:,0], 'bo')
    
    axis.set_xlabel('Chiều cao')
    axis.set_ylabel('Cân nặng')
    axis.set_title('Chiều cao và cân nặng của sinh viên VLU')

    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)
