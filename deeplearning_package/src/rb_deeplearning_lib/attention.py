import numpy as np
from .autogradient import Values 

import numpy as np
from rb_deeplearning_lib import Values # Ensure Values is imported in this scope if the cell were standalone.
                                     # It is imported in u3RHlJRyDl4U.

class Embedding():
  def __init__(self, elements, dims, rangeE=(-1,1)):
    self.elements = elements # Corrected: store elements on self
    self.len = len(elements)
    self.encoder = {el:i for i, el in enumerate(elements)}
    self.decoder = dict(enumerate(elements))
    self.dims = dims

    embeddings = []
    for i in range(self.len):
      embeddings.append(Values((rangeE[0]-rangeE[1])*np.random.rand(dims)+rangeE[1]))
    self.embed = embeddings

  def encode(self, x_elements_flat):
    # x_elements_flat is assumed to be a 1D iterable of elements (e.g., ['a', 'b', 'c'])
    enc = self.encoder
    out = [enc.get(el) for el in x_elements_flat]
    return np.array(out) # Return a numpy array of encoded indices

  def decode(self, y_indices_flat):
    # y_indices_flat is assumed to be a 1D iterable of integer keys
    dec = self.decoder
    out = [dec.get(key) for key in y_indices_flat]
    return out

  def __call__(self, x):
    # Determine the actual data to process and its conceptual shape
    elements_to_process = None
    original_sequence_shape = None # Shape of the input *before* embedding dimension is added

    if isinstance(x, str):
      elements_to_process = list(x) # Treat string as sequence of characters
      original_sequence_shape = (len(x),)
      is_encoded_input = False
    elif isinstance(x, Values):
        x_data = x.vals
        original_sequence_shape = x_data.shape
        elements_to_process = x_data.flatten()
        # Check if elements are already integer indices
        is_encoded_input = np.issubdtype(x_data.dtype, np.integer) or \
                           (x_data.size > 0 and isinstance(elements_to_process[0], int))
    else: # Handles lists, tuples, numpy arrays of elements or integers
        x_data = np.asarray(x)
        original_sequence_shape = x_data.shape
        elements_to_process = x_data.flatten()
        # Check if elements are already integer indices
        is_encoded_input = np.issubdtype(x_data.dtype, np.integer) or \
                           (x_data.size > 0 and isinstance(elements_to_process[0], int))

    if is_encoded_input:
      encoded_indices_flat = elements_to_process
    else:
      encoded_indices_flat = self.encode(elements_to_process)

    num_elements_flat = len(encoded_indices_flat)
    embeded_flat_vals = np.zeros((num_elements_flat, self.dims))

    for i in range(num_elements_flat):
      current_index = encoded_indices_flat[i]
      if current_index is None:
          # Find the original element from elements_to_process at index i
          element_that_failed = elements_to_process[i]
          raise ValueError(f"Element '{element_that_failed}' not found in embedding vocabulary. "
                           f"Vocabulary: {list(self.encoder.keys())}")
      embeded_flat_vals[i,:] = self.embed[current_index].vals

    output_embedding_shape = original_sequence_shape + (self.dims,)
    embeded = Values(embeded_flat_vals.reshape(output_embedding_shape))

    return embeded

  def params(self):
    return self.embed

  def set_params(self, param):
    self.embed = param if isinstance(param, Values) else Values(p)
    
#Modified head for multihead attention
class AttentionHead():
  def __init__(self, dims1,dims2=None,rangeQ=(-1,1),rangeK=(-1,1),rangeV=(-1,1)):
    if not dims2:
      dims2 = dims1

    self.w_q = Values((rangeQ[0]-rangeQ[1])*np.random.rand(dims1,dims2)+rangeQ[1])
    self.w_k = Values((rangeK[0]-rangeK[1])*np.random.rand(dims1,dims2)+rangeK[1])
    self.w_v = Values((rangeV[0]-rangeV[1])*np.random.rand(dims1,dims2)+rangeV[1])
    self.dims = dims2

  def __call__(self, x):
    Q = x @ self.w_q
    K = x @ self.w_k
    V = x @ self.w_v

    axes = list(range(K.vals.ndim))
    axes[-1], axes[-2] = axes[-2], axes[-1]
    
    S = (Q @ K.transpose(tuple(axes)))/(self.dims**0.5)
    A = S.softmax()
    Y = A @ V

    return Y

  def params(self):
    return self.w_q, self.w_k, self.w_v

  def set_params(self, params):
    q, k, v = params
    self.w_q = q if isinstance(q, Values) else Values(q)
    self.w_k = k if isinstance(k, Values) else Values(k)
    self.w_v = v if isinstance(v, Values) else Values(v)

#MultiHead Attention
class AttentionMultiHead:
  def __init__(self, dims, nHeads, rangeQ=(-1,1),rangeK=(-1,1),rangeV=(-1,1), rangeW=(-1,1)):
    head_size = dims//nHeads
    self.heads = [AttentionHead(dims, head_size, rangeQ=rangeQ,rangeK=rangeK,rangeV=rangeV) for _ in range(nHeads)]
    self.w_0 = Values((rangeW[0]-rangeW[1])*np.random.rand(nHeads * head_size, dims)+rangeW[1])
    self.dims = dims
    self.nHeads = nHeads
    self.head_size = head_size

  def __call__(self,x):
    outs = [h(x) for h in self.heads]

    batch_and_seq_dims = x.shape[:-1]
    
    concatenated_feature_dim = self.nHeads * self.head_size
    concatenated_output_vals = np.zeros(batch_and_seq_dims + (concatenated_feature_dim,))
    concatenated_output = Values(concatenated_output_vals)

    current_idx = 0
    for head_output in outs:
      output_width = head_output.shape[-1]
      concatenated_output[..., current_idx : current_idx + output_width] = head_output
      current_idx += output_width
    final_output = concatenated_output @ self.w_0
    return final_output

  def params(self):
    all_params = []
    for h in self.heads:
      all_params.extend(h.params())
    all_params.append(self.w_0)
    return all_params

  def set_params(self, params):
    idx = 0
    for h in self.heads:
      n_head_params = len(self.h.params()) #Should be 3 but just in case
      h.set_params(params[idx:idx+n_head_params])
      idx += n_head_params
    #idx should now be the last value
    if idx != len(params)-1:
      raise ValueError("Mismatched set_params in MultiAttentionHead. Can't set final value.")
    self.w_0 = params[idx] if isinstance(params[idx], Values) else Values(params[idx])
