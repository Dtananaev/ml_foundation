
import numpy as np 
import matplotlib.pyplot as plt 
from linear_regression_helpers import generate_random_samples, alternative_covariance, alternative_variance, mean, plot_linear_regression
from tqdm import tqdm
from optimizers import sgd, sgd_m, sgd_nesterov_m, adagrad, RMSprop, adam, adaBelief, nadam

def grad_b(x,y, y_pred):
    n= len(x)
    d_b = -2/n * sum(x *(y - y_pred))
    return d_b


def grad_a(x,y, y_pred):
    n = len(x)
    d_a = -2/n * sum(y - y_pred)
    return d_a


if __name__ == "__main__":
    num_samples = 25
    min_x, max_x = -2, 2
    print(f"Linear regression with {num_samples} data points")
    # Dataset
    x, y = generate_random_samples(num_samples=num_samples, min_x=min_x, max_x=max_x)
    # Close form solution
    close_form_b = alternative_covariance(x, y)/ alternative_variance(x)
    close_form_a = mean(y) - close_form_b * mean(x)


    a , b = 0.0, 1.0
    lr = 0.1
    num_iter = 1000
    optimizer = adam(lr)
    print(f"Start gradient search with learning rate {lr} and starting a {a}, b {b}, num_iter {num_iter}...")


    xmin, xmax = min_x, max_x
    ymin, ymax = np.min(y), np.max(y)
    
    plt.ion()
    plt.show()
    fig = plt.figure()

    for i in tqdm(range(num_iter), total=num_iter, desc="Optimization"):

        y_pred = a + b* x
        a, b = optimizer.update(np.asarray([a, b]), np.asarray([grad_a(x, y, y_pred), grad_b(x, y, y_pred)]))
        plot_linear_regression(fig, x, y, a, b, xmin, xmax, ymin, ymax)

        plt.pause(0.000001) 
        fig.clf()
    print(f"Done: found a {a} close form {close_form_a} and {b} close form {close_form_b} ...")
    plot_linear_regression(fig, x, y, a, b, xmin, xmax, ymin, ymax)
    ax = fig.add_subplot(1, 2, 1)
    y_pred =close_form_a + close_form_b* x

    ax.plot( x, y_pred, '-b')
    input()

