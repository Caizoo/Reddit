import tensorflow as tf 
import tensorflow.keras as keras 
from tensorflow.keras import layers 
import tensorflow.keras.backend as K 
from tensorflow.python.ops import array_ops

import numpy as np 
import sklearn

def notebook_graph_lstm_test():
    A = np.asarray([[1,1,1,0],[1,1,1,1],[1,1,1,0],[0,1,0,1]])
    x = [[[1,2,3,4],[5,4,6,7],[4,2,5,7],[3,4,5,6]]]

    import numpy as np
    from sklearn.preprocessing import normalize
    N = A.shape[0] # number of nodes in a graph
    D = np.sum(A, 0) # node degrees
    D_hat = np.diag((D + 1e-5)**(-0.5)) # normalized node degrees
    L = np.identity(N) - np.dot(D_hat, A).dot(D_hat) # Laplacian
    Ln = normalize(L, axis=1, norm='l1')
    from scipy.sparse.linalg import eigsh # assumes L to be symmetric
    delta,V = eigsh(Ln,k=20,which='SM') # eigen-decomposition (i.e. find Λ,V)
    print(V)

    V_tf = tf.constant(V, dtype='float32') 
    x_tf = tf.constant(x, dtype='float32')
    g_tf = tf.constant([[np.random.random()] for _ in range(len(x[0]))])


    Vx = tf.matmul(tf.transpose(V_tf), x_tf)
    Vg = tf.matmul(tf.transpose(V_tf), g_tf)
    out = tf.matmul(V_tf, tf.multiply(Vx[0], Vg[0]))

    print(out)

class LSTMCai(layers.LSTMCell):

    def __init__(self, units, **kwargs):
        super(LSTMCai, self).__init__(units, **kwargs)

    def _compute_carry_and_output(self, x, h_tm1, c_tm1):
        """Computes carry and output using split kernels."""
        x_i, x_f, x_c, x_o = x
        h_tm1_i, h_tm1_f, h_tm1_c, h_tm1_o = h_tm1
        i = self.recurrent_activation(
            x_i + K.dot(h_tm1_i, self.recurrent_kernel[:, :self.units]))
        f = self.recurrent_activation(x_f + K.dot(
            h_tm1_f, self.recurrent_kernel[:, self.units:self.units * 2]))
        c = f * c_tm1 + i * self.activation(x_c + K.dot(
            h_tm1_c, self.recurrent_kernel[:, self.units * 2:self.units * 3]))
        o = self.recurrent_activation(
            x_o + K.dot(h_tm1_o, self.recurrent_kernel[:, self.units * 3:]))
        return c, o

    def call(self, inputs, states, training=None):
        h_tm1 = states[0]  # previous memory state
        c_tm1 = states[1]  # previous carry state

        dp_mask = self.get_dropout_mask_for_cell(inputs, training, count=4)
        rec_dp_mask = self.get_recurrent_dropout_mask_for_cell(
            h_tm1, training, count=4)

        
        if 0 < self.dropout < 1.:
            inputs_i = inputs * dp_mask[0]
            inputs_f = inputs * dp_mask[1]
            inputs_c = inputs * dp_mask[2]
            inputs_o = inputs * dp_mask[3]
        else:
            inputs_i = inputs
            inputs_f = inputs
            inputs_c = inputs
            inputs_o = inputs
        k_i, k_f, k_c, k_o = array_ops.split(
          self.kernel, num_or_size_splits=4, axis=1)
        x_i = K.dot(inputs_i, k_i)
        x_f = K.dot(inputs_f, k_f)
        x_c = K.dot(inputs_c, k_c)
        x_o = K.dot(inputs_o, k_o)
        if self.use_bias:
            b_i, b_f, b_c, b_o = array_ops.split(
                self.bias, num_or_size_splits=4, axis=0)
            x_i = K.bias_add(x_i, b_i)
            x_f = K.bias_add(x_f, b_f)
            x_c = K.bias_add(x_c, b_c)
            x_o = K.bias_add(x_o, b_o)

        if 0 < self.recurrent_dropout < 1.:
            h_tm1_i = h_tm1 * rec_dp_mask[0]
            h_tm1_f = h_tm1 * rec_dp_mask[1]
            h_tm1_c = h_tm1 * rec_dp_mask[2]
            h_tm1_o = h_tm1 * rec_dp_mask[3]
        else:
            h_tm1_i = h_tm1
            h_tm1_f = h_tm1
            h_tm1_c = h_tm1
            h_tm1_o = h_tm1
        x = (x_i, x_f, x_c, x_o)
        h_tm1 = (h_tm1_i, h_tm1_f, h_tm1_c, h_tm1_o)
        c, o = self._compute_carry_and_output(x, h_tm1, c_tm1)

        h = o * self.activation(c)
        return h, [h, c]
    
class LSTMCai(layers.LSTMCell):

    def __init__(self, units, **kwargs):
        super(LSTMCai, self).__init__(units, **kwargs)

    def _compute_carry_and_output(self, x, h_tm1, c_tm1):
        """Computes carry and output using split kernels."""
        x_i, x_f, x_c, x_o = x
        h_tm1_i, h_tm1_f, h_tm1_c, h_tm1_o = h_tm1
        i = self.recurrent_activation(
            x_i + K.dot(h_tm1_i, self.recurrent_kernel[:, :self.units]))
        f = self.recurrent_activation(x_f + K.dot(
            h_tm1_f, self.recurrent_kernel[:, self.units:self.units * 2]))
        c = f * c_tm1 + i * self.activation(x_c + K.dot(
            h_tm1_c, self.recurrent_kernel[:, self.units * 2:self.units * 3]))
        o = self.recurrent_activation(
            x_o + K.dot(h_tm1_o, self.recurrent_kernel[:, self.units * 3:]))
        return c, o

    def call(self, inputs, states, training=None):
        h_tm1 = states[0]  # previous memory state
        c_tm1 = states[1]  # previous carry state

        dp_mask = self.get_dropout_mask_for_cell(inputs, training, count=4)
        rec_dp_mask = self.get_recurrent_dropout_mask_for_cell(
            h_tm1, training, count=4)

        
        if 0 < self.dropout < 1.:
            inputs_i = inputs * dp_mask[0]
            inputs_f = inputs * dp_mask[1]
            inputs_c = inputs * dp_mask[2]
            inputs_o = inputs * dp_mask[3]
        else:
            inputs_i = inputs
            inputs_f = inputs
            inputs_c = inputs
            inputs_o = inputs
        k_i, k_f, k_c, k_o = array_ops.split(
          self.kernel, num_or_size_splits=4, axis=1)
        x_i = K.dot(inputs_i, k_i)
        x_f = K.dot(inputs_f, k_f)
        x_c = K.dot(inputs_c, k_c)
        x_o = K.dot(inputs_o, k_o)
        if self.use_bias:
            b_i, b_f, b_c, b_o = array_ops.split(
                self.bias, num_or_size_splits=4, axis=0)
            x_i = K.bias_add(x_i, b_i)
            x_f = K.bias_add(x_f, b_f)
            x_c = K.bias_add(x_c, b_c)
            x_o = K.bias_add(x_o, b_o)

        if 0 < self.recurrent_dropout < 1.:
            h_tm1_i = h_tm1 * rec_dp_mask[0]
            h_tm1_f = h_tm1 * rec_dp_mask[1]
            h_tm1_c = h_tm1 * rec_dp_mask[2]
            h_tm1_o = h_tm1 * rec_dp_mask[3]
        else:
            h_tm1_i = h_tm1
            h_tm1_f = h_tm1
            h_tm1_c = h_tm1
            h_tm1_o = h_tm1
        x = (x_i, x_f, x_c, x_o)
        h_tm1 = (h_tm1_i, h_tm1_f, h_tm1_c, h_tm1_o)
        c, o = self._compute_carry_and_output(x, h_tm1, c_tm1)

        h = o * self.activation(c)
        return h, [h, c]
    
    
def test_graph():
    A = np.asarray([[1,1,1,0],[1,1,1,1],[1,1,1,0],[0,1,0,1]])
    
    xs = [[1,3,3,2], [2,2,2,1], [2,3,4,2]]
    x = [[1],[3],[3],[2]]
    g = [[np.random.random() for _ in range(8)] for _ in range(4)]
    N = A.shape[0] # number of nodes in a graph
    D = np.sum(A, 0) # node degrees
    D_hat = np.diag((D + 1e-5)**(-0.5)) # normalized node degrees
    L = np.identity(N) - np.dot(D_hat, A).dot(D_hat) # Laplacian
    Ln = sklearn.preprocessing.normalize(L, axis=1, norm='l1')
    from scipy.sparse.linalg import eigsh # assumes L to be symmetric
    delta,V = eigsh(Ln,k=20,which='SM') # eigen-decomposition (i.e. find Λ,V)
    print(V)

    V_tf = tf.constant(V, dtype='float32') 
    x_tf = tf.constant(x, dtype='float32')
    g_tf = tf.constant(g, dtype='float32')
    

    Vx = tf.matmul(tf.transpose(V_tf), x_tf)
    Vg = tf.matmul(tf.transpose(V_tf), g_tf)
    out = tf.matmul(V_tf, tf.multiply(Vx, Vg))

    print(out)
    
    
test_graph()