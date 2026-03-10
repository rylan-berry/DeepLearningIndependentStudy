from .autogradient import Values

class FullRNNLayer:
  def __init__(self, input_dim, hidden_dim, output_dim, rangeW=(0,1),rangeB=(0,1), activ="tanh", transpInput=False):
    self.hidd_dim = hidden_dim
    self.out_dim = output_dim

    self.W_xh = Values((rangeW[0]-rangeW[1])*np.random.rand(input_dim, hidden_dim)+rangeW[1])
    self.W_hh = Values((rangeW[0]-rangeW[1])*np.random.rand(hidden_dim, hidden_dim)+rangeW[1])
    self.b_h = Values((rangeW[0]-rangeW[1])*np.random.rand(1,hidden_dim)+rangeW[1])
    self.W_hy = Values((rangeW[0]-rangeW[1])*np.random.rand(hidden_dim,output_dim)+rangeW[1])
    self.b_y = Values((rangeW[0]-rangeW[1])*np.random.rand(1,output_dim)+rangeW[1])
    self.activ = activ
    self.transpInput = transpInput

  def _rnn_cell_forward(self, x_t, h_prev):
    # Corrected 'self.active' to 'self.activ' and added function call '()'
    # If self.activation is not a valid method name, it will raise an AttributeError.

    #x_t ~ (B, D); h_t ~ (B,H); y_t ~ (B,O);
    h_t = getattr((x_t @ self.W_xh + h_prev @self.W_hh + self.b_h), self.activ)()
    y_t = h_t @ self.W_hy + self.b_y
    return h_t, y_t


  def __call__(self,x): #Assumed first dimension is time dimension, assumed final dimension is batch dimension
    x_n = x
    if self.transpInput: #assuming (B,T,D) and converting to (T,B,D)
      axis = list(range(len(x.shape))) #create list from 0 to total input dims
      axis[0]=1;axis[1]=0

      x_n = x.transpose(axis)
    x_siz = x_n.vals.shape # Use .vals to get numpy shape for Values object
    # Initial hidden state needs to match batch dimension of inputs
    h_t = Values(np.zeros((x_siz[1],self.hidd_dim, ))) # x_siz[1] will be batch_size if x_n is (Time, Batch, Data)


    h = []
    all_y_ts = Values(np.zeros((x_siz[0],x_siz[1], self.out_dim))) # Should collect as a Values variable and not a list, should output as (T, B, O)

    for i, x_t in enumerate(x_n):
        h_t, y_t = self._rnn_cell_forward(x_t, h_t)
        h.append(h_t)
        all_y_ts[i] = y_t

    fin_out = all_y_ts
    if self.transpInput:
      axis = list(range(len(x.shape)))
      axis[0]=1;axis[1]=0

      fin_out = all_y_ts.transpose(axis)

    return fin_out

  def params(self):
    return self.W_xh, self.W_hh, self.b_h, self.W_hy, self.b_y

#I want to also make a partial RNN layer(s) so things can be changed around it -- may not do this because of time.
#Change the output to be a Values that we can set the items of. then make the output fill that in
