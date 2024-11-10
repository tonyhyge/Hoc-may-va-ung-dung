from flask import Flask, Response
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from cvxopt import matrix as cvxopt_matrix
from cvxopt import solvers as cvxopt_solvers
import matplotlib
matplotlib.use("SVG")

app = Flask(__name__)

@app.route('/')
def soft_margin_svm_plot():
    # Training data
    x = np.array([[0.2, 0.869], [0.687, 0.212], [0.822, 0.411], [0.738, 0.694], 
                  [0.176, 0.458], [0.306, 0.753], [0.936, 0.413], [0.215, 0.410], 
                  [0.612, 0.375], [0.784, 0.602], [0.612, 0.554], [0.357, 0.254], 
                  [0.204, 0.775], [0.512, 0.745], [0.498, 0.287], [0.251, 0.557], 
                  [0.502, 0.523], [0.119, 0.687], [0.495, 0.924], [0.612, 0.851]])
    y = np.array([-1, 1, 1, 1, -1, -1, 1, -1, 1, 1, -1, 1, -1, 1, -1, -1, 1, -1, 1, 1])
    y = y.astype('float').reshape(-1, 1)

    # Solve QP problem
    C = 50.0
    N = x.shape[0]
    H = (y @ y.T) * (x @ x.T)
    P = cvxopt_matrix(H)
    q = cvxopt_matrix(np.ones(N) * -1)
    A = cvxopt_matrix(y.reshape(1, -1))
    b = cvxopt_matrix(np.zeros(1))
    g = np.vstack([-np.eye(N), np.eye(N)])
    G = cvxopt_matrix(g)
    h1 = np.hstack([np.zeros(N), np.ones(N) * C])
    h = cvxopt_matrix(h1)

    cvxopt_solvers.options['abstol'] = 1e-10
    cvxopt_solvers.options['reltol'] = 1e-10
    cvxopt_solvers.options['feastol'] = 1e-10
    cvxopt_solvers.options['maxiters'] = 1000
    cvxopt_solvers.options['show_progress'] = True

    sol = cvxopt_solvers.qp(P, q, G, h, A, b)
    lamb = np.array(sol['x'])

    # Calculate w and b
    w = np.sum(lamb * y * x, axis=0)

    sv_idx = np.where(lamb > 1e-5)[0]
    sv_lamb = lamb[sv_idx]

    sv_x = x[sv_idx]
    sv_y = y[sv_idx]
    b = sv_y[0] - np.dot(sv_x[0], w)

    y_hat = np.dot(x, w) + b
    slack = np.maximum(0, 1 - y.flatten() * y_hat)
    total_slack = np.sum(slack)

    plt.figure(figsize=(7, 7))
    color = ['red' if a == 1 else 'blue' for a in y]
    plt.scatter(x[:, 0], x[:, 1], s=200, c=color, alpha=0.7)
    plt.xlim(0, 1)
    plt.ylim(0, 1)

    # Decision boundary
    x1_dec = np.linspace(0, 1, 200)
    x2_dec = (-b - w[0] * x1_dec) / w[1] #(-w[0] * x1_dec + b) / w[1]
    plt.plot(x1_dec, x2_dec, c='black', lw=1.0, label='decision boundary')



    for s, (x1, x2) in zip(slack, x):
        plt.annotate(str(s.round(2)), (x1-0.02, x2 + 0.03))

    # Visualize the positive & negative boundary and support vectors
    w_norm = np.sqrt(np.sum(w ** 2))
    w_unit = w / w_norm
    half_margin = 1 / w_norm

    upper = np.vstack([x1_dec, x2_dec]).T + half_margin * w_unit
    lower = np.vstack([x1_dec, x2_dec]).T - half_margin * w_unit

    plt.plot(upper[:, 0], upper[:, 1], '--', lw = 1.0, label = 'positive boundary')
    plt.plot(lower[:, 0], lower[:, 1], '--', lw = 1.0, label = 'negative boundary')
        
    plt.scatter(sv_x[:, 0], sv_x[:, 1], s=60, marker='o', c='white')
    plt.legend()
    plt.title('C = ' + str(C) + ',  Σξ = ' + str(np.sum(total_slack).round(2)))

    # Convert plot to PNG image for display in Flask
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()

    return Response(img.getvalue(), mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)
