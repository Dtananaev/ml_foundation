
import numpy as np

class sgd:
    
    def __init__(self, lr):
        self.lr = lr
    
    def update(self, variable, grad):
        return variable - self.lr * grad


class sgd_m:
    def __init__(self, lr, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None
    
    def update(self, variable, grad):
        if self.v is None:
            self.v = np.zeros_like(variable)
        self.v =  self.momentum * self.v - self.lr * grad
        return variable + self.v



class sgd_nesterov_m:
    def __init__(self, lr, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None
    
    def update(self, variable, grad):
        if self.v is None:
            self.v = np.zeros_like(variable)
        v_prev = self.v
        self.v = self.momentum * self.v - self.lr * grad

        return variable - self.momentum * v_prev + (1 + self.momentum) *self.v

class adagrad:
    def __init__(self, lr):
        self.lr = lr
        self.cache = None

    def update(self, variable, grad):
        if self.cache is None:
            self.cache = np.zeros_like(variable)
        self.cache += grad **2
        return variable - self.lr * grad / (np.sqrt(self.cache) + 1e-7) 


class RMSprop:
    def __init__(self, lr, cache_decay=0.9):
        self.lr = lr
        self.cache_decay = cache_decay
        self.cache = None

    def update(self, variable, grad):
        if self.cache is None:
            self.cache = np.zeros_like(variable)
        self.cache = self.cache_decay * self.cache + (1.0- self.cache_decay) * grad **2
        return variable - self.lr * grad / (np.sqrt(self.cache) + 1e-7) 

class adam:

    def __init__(self, lr, beta1=0.9, beta2=0.999):
        self.lr = lr
        self.beta1= beta1
        self.beta2 = beta2
        self.m = None
        self.v = None
        self.t = 0.0

    def update(self, variable, grad):
        self.t += 1.0

        if self.m is None:
            self.m = np.zeros_like(variable)
        if self.v is None:
            self.v = np.zeros_like(variable)
        self.m = self.beta1 * self.m + (1.0 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1.0 - self.beta2) * (grad ** 2)
        mb = self.m /(1.0 - self.beta1 ** self.t)
        vb = self.v / (1.0 - self.beta2 ** self.t)

        return variable - self.lr * mb / (np.sqrt(vb)+ 1e-7)


class nadam:

    def __init__(self, lr, beta1=0.9, beta2=0.999):
        self.lr = lr
        self.beta1= beta1
        self.beta2 = beta2
        self.m = None
        self.v = None
        self.nesterov_result = None
        self.t = 0.0

    def update(self, variable, grad):
        self.t += 1.0
        if self.m is None:
            self.m = np.zeros_like(variable)
        if self.v is None:
            self.v = np.zeros_like(variable)
        if self.nesterov_result is None:
             self.nesterov_result = np.zeros_like(variable)


        self.m = self.beta1 * self.m + (1.0 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1.0 - self.beta2) * (grad ** 2)
        mb = self.m /(1.0 - self.beta1 ** self.t)
        vb = self.v / (1.0 - self.beta2 ** self.t)

        nest_g =  (self.beta1 * mb + (1.0 - self.beta1) * grad ) / (1.0 - self.beta1 ** self.t)
 
        return variable - self.lr * nest_g / (np.sqrt(vb)+ 1e-7)


class adaBelief:

    def __init__(self, lr, beta1=0.9, beta2=0.999):
        self.lr = lr
        self.beta1= beta1
        self.beta2 = beta2
        self.m = None
        self.v = None
        self.t = 0.0

    def update(self, variable, grad):

        self.t += 1.0
        if self.m is None:
            self.m = np.zeros_like(variable)
        if self.v is None:
            self.v = np.zeros_like(variable)
        self.m = self.beta1 * self.m + (1.0 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1.0 - self.beta2) * ((grad - self.m)** 2) + 1e-7
        mb = self.m /(1.0 - self.beta1 ** self.t)
        vb = self.v / (1.0 - self.beta2 ** self.t)
        return variable - self.lr * mb / (np.sqrt(vb)+ 1e-7)