from autogradient import Values
import numpy as np
class Optimizer:
  def __init__(self):
    pass

  def step(self, params, learning_rate):
    for p in params:
      p.vals = p.vals - learning_rate * p.grad
      p.grad = np.zeros_like(p.grad)

class Optim_SGD(Optimizer):
  def __init__(self, finitters, fin_l_rate):
    self.t = 0
    self.finitter = finitters
    self.fin_l_rate = fin_l_rate

  def step(self, params, learning_rate):
    self.t += 1
    t = self.t
    alpha = t/self.finitters
    if(alpha < 1):
      l_rate = learning_rate*(1-alpha) + alpha*self.fin_l_rate
    else:
      l_rate = self.fin_l_rate
    for p in params:
      p.vals = p.vals - l_rate * p.grad
      p.grad = np.zeros_like(p.grad)

class Optim_SGD_Momentum(Optimizer):
  def __init__(self,mom_beta=0.9):
    self.v = {}
    self.beta = mom_beta

  def step(self, params, learning_rate):
    v = self.v
    for p in params:
      if p not in self.v:
        v[p] = np.zeros_like(p.vals)
      v[p] = self.beta*v[p] - learning_rate*p.grad
      p.vals = p.vals + self.v[p]
      p.grad = np.zeros_like(p.grad)
    self.v = v

class Optim_AdaGrad(Optimizer):
  def __init__(self, gamma=0.0000001):
    self.gamma = gamma
    self.r = {}

  def step(self, params, l_rate):
    for p in params:
      if p not in self.r:
        self.r[p] = np.zeros_like(p.vals)
      self.r[p] = self.r[p] + p.grad**2
      p.vals = p.vals - l_rate * p.grad / (self.gamma + self.r[p]**0.5)
      p.grad = np.zeros_like(p.grad)

class Optim_RMSPropclass(Optimizer):
  def __init__(self,decay_rate, gamma=0.000001):
    self.decay_rate = decay_rate
    self.gamma = gamma
    self.r = {}
  def step(self, params, l_rate):
    dr = self.decay_rate
    for p in params:
      if p not in self.r:
        self.r[p] = np.zeros_like(p.vals)
      self.r[p] = dr*self.r[p] + (1-dr)*p.grad**2
      p.vals = p.vals - l_rate*p.grad/(self.gamma + self.r[p]**0.5)
      p.grad = np.zeros_like(p.grad)

class Optim_Adam(Optimizer):
  def __init__(self, beta1, beta2, gamma = 0.000001):
    self.b1 = beta1
    self.b2 = beta2
    self.gamma = gamma
    self.r = {}
    self.s = {}
    self.t = 0

  def step(self, params, l_rate):
    self.t += 1
    t = self.t
    beta1 = self.b1
    beta2 = self.b2
    for p in params:
      if p not in self.r:
        self.r[p] = np.zeros_like(p.vals)
      if p not in self.s:
        self.s[p] = np.zeros_like(p.vals)

      self.s[p] = beta1*self.s[p] + (1-beta1)*p.grad
      self.r[p] = beta2*self.r[p] + (1-beta2)*p.grad**2

      s_hat = self.s[p]/(1-beta1**t)
      r_hat = self.r[p]/(1-beta2**t)

      p.vals = p.vals - l_rate*s_hat/(self.gamma + r_hat**0.5)
      p.grad = np.zeros_like(p.grad)
