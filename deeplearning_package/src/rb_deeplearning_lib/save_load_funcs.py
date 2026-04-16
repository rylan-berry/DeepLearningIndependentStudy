import numpy as np
from .autogradient import Values

#used to convert the params saved by the model to a python list that can be saved in any desired format.
def getModelSave(model):
  params = model.blocks.params()
  simple_params = []
  for p in params:
    simple_params.append(p.vals.tolist())
  return simple_params

#Converts lists of parameter matricies (can be either a numpy array or list)
def modelLoadParams(params, model):
  if isinstance(params[0],np.ndarray) or isinstance(params[0], Values):
    model.set_params(params)
  else:
    params2 = []
    for p in params:
      params2.append(Values(np.array(p)))
    model.set_params(params2)
  return model

