import numpy as np
from .autogradient import Values 

#Grouped with attention so values can be embeded.
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
    # Ensure x_data is a numpy array for consistent handling
    if isinstance(x, Values):
        x_data = x.vals
    else:
        # Attempt to convert to numpy array. This handles lists, tuples, etc.
        # If x is a single string like 'abcd', np.asarray('abcd') will result in a 0-dim array.
        # If x is a list of strings ['a', 'b'], it becomes a 1-dim array.
        x_data = np.asarray(x)

    # Store the original shape of the input data (before potential flattening)
    original_input_shape = x_data.shape

    # Determine if x_data contains already encoded integers or elements to be encoded
    is_encoded_input = False
    if np.issubdtype(x_data.dtype, np.integer):
      is_encoded_input = True
    elif x_data.size > 0: # Check if array is not empty
      # Check type of first element if not clearly integer dtype
      # Reshape to -1 to get a 1D view, then check the type of the first item.
      if isinstance(x_data.reshape(-1)[0], int):
          is_encoded_input = True

    if is_encoded_input:
      encoded_indices_flat = x_data.flatten()
    else:
      # Flatten the input elements to pass to the encode method
      x_elements_flat = x_data.flatten()
      encoded_indices_flat = self.encode(x_elements_flat)

    # Now, encoded_indices_flat is a 1D numpy array of integer indices
    # Create the output embeddings
    num_elements_flat = len(encoded_indices_flat)
    embeded_flat_vals = np.zeros((num_elements_flat, self.dims))

    for i in range(num_elements_flat):
      # Access the underlying numpy array of the Values object in self.embed
      embeded_flat_vals[i,:] = self.embed[encoded_indices_flat[i]].vals

    # Reshape the flat embeddings back to the desired output shape
    # The output shape will be (original_input_shape, self.dims)
    output_embedding_shape = original_input_shape + (self.dims,)
    embeded = Values(embeded_flat_vals.reshape(output_embedding_shape))

    return embeded

  def params(self):
    return self.embed
    
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

    S = (Q @ K.transpose((-1, -2)))/(self.dims**0.5)
    A = S.softmax()
    Y = A @ V

    return Y
  
  def params(self):
    return self.w_q, self.w_k, self.w_v

#MultiHead Attention
class AttentionMultiHead:
  def __init__(self, dims, nHeads, rangeQ=(-1,1),rangeK=(-1,1),rangeV=(-1,1), rangeW=(-1,1)):
    head_size = dims//nHeads
    self.heads = [AttentionHead(dims, head_size, rangeQ=(-1,1),rangeK=(-1,1),rangeV=(-1,1)) for _ in range(nHeads)]
    # w_0 should project from (nHeads * head_size) to dims
    self.w_0 = Values((rangeW[0]-rangeW[1])*np.random.rand(nHeads * head_size, dims)+rangeW[1])
    self.dims = dims # Store dims for potential future use
    self.nHeads = nHeads # Store nHeads
    self.head_size = head_size # Store head_size

  def __call__(self,x):
    outs = [h(x) for h in self.heads]

    seq_len = x.shape[0]
    # Create a Values object to hold the concatenated result
    # Initializing with zeros to ensure a valid array for slice assignment
    concatenated_output = Values(np.zeros((seq_len, self.nHeads * self.head_size)))

    current_idx = 0
    for head_output in outs:
      output_width = head_output.shape[1] # Should be self.head_size
      # Assign the head's output to the corresponding slice of the concatenated_output Values object
      concatenated_output[:, current_idx : current_idx + output_width] = head_output
      current_idx += output_width

    # Apply the final linear layer (projection) using w_0
    final_output = concatenated_output @ self.w_0
    return final_output

  def params(self):
    all_params = []
    for h in self.heads:
      all_params.extend(h.params())
    all_params.append(self.w_0)
    return all_params
