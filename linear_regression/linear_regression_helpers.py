
import numpy as np
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

def generate_random_samples(num_samples, min_x=-5, max_x=5, noise=5):
    """Generates random points for linear regression."""

    a =  -12 * (2 * np.random.rand(1) -1.)
    b =  5* (2 * np.random.rand(1) -1.)
    x = (np.random.rand(num_samples) + min_x) * (max_x - min_x)
    y =  a + b * x + np.random.rand(num_samples)* noise

    return x, y


def mean(x):
    """Mean"""
    n = len(x)
    return sum(x) / n 


def variance(x):
    """Variance"""
    n = len(x)
    mu_x = mean(x)
    var =(x - mu_x)**2

    return  sum(var) / n 


def alternative_variance(x):
    """var = mean(x**2) - mean(x)**2 """
    return mean(x**2) - mean(x)**2


def alternative_covariance(x,y):
    """ """
    return mean(x*y) - mean(x) * mean(y)


def plot_linear_regression(fig, x, y, a, b, xmin, xmax, ymin, ymax):

    def error(a, b, x, y):
        return sum((y - (b * x + a)) ** 2) / float(len(x))
    y_pred = b * x + a 

    ax = fig.add_subplot(1, 2, 1)
    ax.plot(x,y, 'go', x,y_pred, '-r')
    #ax.set_xlim([xmin, xmax])
    #ax.set_ylim([0.5* ymin, 2*ymax])
    ax = fig.add_subplot(1, 2, 2, projection='3d')
    a_s = np.linspace(-5.0, 5.0, 10)
    b_s = np.linspace(-5.0, 5.0, 10)

    A, B = np.meshgrid(a_s, b_s)
    zs = np.array([error(a_s, b_s, x, y) 
                for a_s, b_s in zip(np.ravel(A), np.ravel(B))])
    Z = zs.reshape(A.shape)


    ax.plot_surface( B, A, Z, rstride=1, cstride=1, color='b', alpha=0.5)
    ax.set_xlabel('b')
    ax.set_ylabel('a')
    ax.set_zlabel('error')

    ax.scatter(b, a, error(a, b, x, y), color="r", s=20)
    return fig