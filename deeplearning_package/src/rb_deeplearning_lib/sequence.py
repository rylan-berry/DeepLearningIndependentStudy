from .autogradient import Values

class Sequence:
  def __init__(self, arr):
    self.arr = arr

  def __call__(self, x):
    x_i = x
    for item in self.arr:
      x_i = item(x_i)
    return x_i

  def params(self):
    all_params = []
    for l in self.arr:
      # Check if the item has a params method (e.g., Layer or Dense)
      if hasattr(l, 'params'):
        # layer_params should return weights and biases, which can be individual Values or lists of Values
        layer_params = l.params()
        # Ensure layer_params is iterable (e.g., a tuple of (weights, biases))
        if isinstance(layer_params, tuple) or isinstance(layer_params, list):
            for p_group in layer_params:
                # If p_group is a list (e.g., from Dense containing multiple layers),
                # extend with its elements, otherwise append the Value object itself.
                if isinstance(p_group, list):
                    all_params.extend(p_group)
                else:
                    all_params.append(p_group)
        elif isinstance(layer_params, Values):
          all_params.append(layer_params):
        

    return all_params
