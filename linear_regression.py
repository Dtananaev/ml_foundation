import numpy as np 
import matplotlib.pyplot as plt 



def generate_regression_samples(num_samples, min_x=0, max_x=100, a=12, b=1.2, e=10):

    x = (np.random.rand(num_samples) + min_x) * (max_x - min_x)
    y=  a + b* x + np.random.rand(num_samples)* e
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

    return  sum(var)/n 

def alternative_variance(x):
    """var = mean(x**2) - mean(x)**2 """
    return mean(x**2) - mean(x)**2


def alternative_covariance(x,y):
    return mean(x*y) - mean(x) * mean(y) 

print("Linear regression")
num_samples = 50
min_x, max_x = 0, 100
x, y = generate_regression_samples(num_samples=num_samples)
b = alternative_covariance(x,y)/ alternative_variance(x)
a = mean(y) - b * mean(x)


poly1d_fn = lambda x: a + b * x
#plt.scatter(x,y)
plt.plot(x,y, 'go', x, poly1d_fn(x), '-r') 




plt.show()

print(f"x {x} mean {mean(x)} variance {variance(x)} alt var {alternative_variance(x)}")