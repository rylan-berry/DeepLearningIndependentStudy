import numpy as np
from .autogradient import Values 

#Grouped with attention so values can be embeded.
class Embedding():
  def __init__(self, elements, dims, rangeE=(-1,1)):
    elements=elements
    len_e = len(elements)
    self.len = len_e
    self.encoder = {el:i for i, el in enumerate(elements)}
    self.decoder = dict(enumerate(elements))
    self.dims = dims # Store dims for use in __call__
    
    embeddings = []
    for i in range(len_e):
      embeddings.append(Values((rangeE[0]-rangeE[1])*np.random.rand(dims)+rangeE[1]))

    self.embed = embeddings

  def encode(self, x):
    #Assuming x is an array of the elements
    enc = self.encoder
    out = [enc.get(el) for el in x]
    return out
  def decode(self, y):
    #Assuming x is an array of keys
    dec = self.decoder
    out = [dec.get(key) for key in y]
    return out

  def __call__(self, x):
    x=x
    l_x = len(x)
    #assumed that an integer input is already encoded
    encoded = []
    if isinstance(x[0],int):
      encoded = x
    else:
      encoded = self.encode(x)
    embeded = Values(np.zeros((l_x,self.dims))) 
    emb = self.embed
    for i in range(l_x):
      embeded[i,:]=emb[encoded[i]]
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
