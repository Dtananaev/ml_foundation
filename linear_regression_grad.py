import numpy as np 
import matplotlib.pyplot as plt 



def generate_regression_samples(num_samples, min_x=0, max_x=100, a=-12, b=-1.2, e=10):

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



def grad_b(x,y, y_pred):
    n= len(x)
    d_b = -2/n * sum(x *(y - y_pred))
    return d_b


def grad_a(x,y, y_pred):
    n = len(x)
    d_a = -2/n * sum(y - y_pred)
    return d_a
print("Linear regression")
num_samples = 50
min_x, max_x = 0, 100
x, y = generate_regression_samples(num_samples=num_samples)

a = 0.0
b = 0.0
lr = 0.00001
num_iter = 1000
plt.ion()
plt.show()
p = []
ax = plt.gca()
ax.set_xlim([min_x, max_x])
#ax.set_ylim([0 , 200])
for i in range(num_iter):

 
    y_pred = a + b* x
    #plt.scatter(x,y)
    plt.plot(x,y, 'go', x,y_pred, '-r')
    a = a - lr * grad_a(x, y, y_pred)
    b = b - lr * grad_b(x,y, y_pred)
    input()
    plt.pause(0.1)
    plt.clf()


plt.plot(x,y, 'go', x, poly1d_fn(x), '-r')
plt.show()

print(f"x {x} mean {mean(x)} variance {variance(x)} alt var {alternative_variance(x)}")