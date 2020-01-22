from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.seq2seq.python.ops.attention_wrapper import BahdanauAttention
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_scope
import tensorflow as tf
import math
from Utils.Utils import shape_list
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.contrib.seq2seq.python.ops import attention_wrapper
from functools import partial



class GmmAttention(attention_wrapper.AttentionMechanism):
    def __init__(self,
                 memory,
                 num_mixtures=16,
                 memory_sequence_length=None,
                 check_inner_dims_defined=True,
                 score_mask_value=None,
                 name='GmmAttention'):
        self.dtype = memory.dtype
        self.num_mixtures = num_mixtures
        self.query_layer = tf.layers.Dense(
            3 * num_mixtures, name='gmm_query_layer', use_bias=True, dtype=self.dtype)

        with tf.name_scope(name, 'GmmAttentionMechanismInit'):
            if score_mask_value is None:
                score_mask_value = 0.
            self._maybe_mask_score = partial(
                attention_wrapper._maybe_mask_score,
                memory_sequence_length=memory_sequence_length,
                score_mask_value=score_mask_value)
            self._value = attention_wrapper._prepare_memory(
                memory, memory_sequence_length, check_inner_dims_defined)
            self._batch_size = (
                    self._value.shape[0].value or tf.shape(self._value)[0])
            self._alignments_size = (
                    self._value.shape[1].value or tf.shape(self._value)[1])

    @property
    def values(self):
        return self._value

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def alignments_size(self):
        return self._alignments_size

    @property
    def state_size(self):
        return self.num_mixtures

    def initial_alignments(self, batch_size, dtype):
        max_time = self._alignments_size
        return rnn_cell_impl._zero_state_tensors(max_time, batch_size, dtype)

    def initial_state(self, batch_size, dtype):
        state_size_ = self.state_size
        return rnn_cell_impl._zero_state_tensors(state_size_, batch_size, dtype)

    def __call__(self, query, state):
        with tf.variable_scope("GmmAttention"):
            previous_kappa = state

            params = self.query_layer(query)
            alpha_hat, beta_hat, kappa_hat = tf.split(params, num_or_size_splits=3, axis=1)

            # [batch_size, num_mixtures, 1]
            alpha = tf.expand_dims(tf.exp(alpha_hat), axis=2)
            # softmax makes the alpha value more stable.
            # alpha = tf.expand_dims(tf.nn.softmax(alpha_hat, axis=1), axis=2)
            beta = tf.expand_dims(tf.exp(beta_hat), axis=2)
            kappa = tf.expand_dims(previous_kappa + tf.exp(kappa_hat), axis=2)

            # [1, 1, max_input_steps]
            mu = tf.reshape(tf.cast(tf.range(self.alignments_size), dtype=tf.float32),
                            shape=[1, 1, self.alignments_size])

            # [batch_size, max_input_steps]
            phi = tf.reduce_sum(alpha * tf.exp(-beta * (kappa - mu) ** 2.), axis=1)

        alignments = self._maybe_mask_score(phi)
        state = tf.squeeze(kappa, axis=2)

        return alignments, state


class LocationSensitiveAttention(BahdanauAttention):
    """Impelements Bahdanau-style (cumulative) scoring function.
    Usually referred to as "hybrid" attention (content-based + location-based)
    Extends the additive attention described in:
    "D. Bahdanau, K. Cho, and Y. Bengio, “Neural machine transla-
    tion by jointly learning to align and translate,” in Proceedings
    of ICLR, 2015."
    to use previous alignments as additional location features.

    This attention is described in:
    J. K. Chorowski, D. Bahdanau, D. Serdyuk, K. Cho, and Y. Ben-
    gio, “Attention-based modules for speech recognition,” in Ad-
    vances in Neural Information Processing Systems, 2015, pp.
    577–585.
    """

    def __init__(self,
                 num_units,
                 memory,
                 hparams,
                 mask_encoder=True,
                 memory_sequence_length=None,
                 smoothing=False,
                 cumulate_weights=True,
                 name='LocationSensitiveAttention'):
        """Construct the Attention mechanism.
        Args:
            num_units: The depth of the query mechanism.
            memory: The memory to query; usually the output of an RNN encoder.  This
                tensor should be shaped `[batch_size, max_time, ...]`.
            mask_encoder (optional): Boolean, whether to mask encoder paddings.
            memory_sequence_length (optional): Sequence lengths for the batch entries
                in memory.  If provided, the memory tensor rows are masked with zeros
                for values past the respective sequence lengths. Only relevant if mask_encoder = True.
            smoothing (optional): Boolean. Determines which normalization function to use.
                Default normalization function (probablity_fn) is softmax. If smoothing is
                enabled, we replace softmax with:
                        a_{i, j} = sigmoid(e_{i, j}) / sum_j(sigmoid(e_{i, j}))
                Introduced in:
                    J. K. Chorowski, D. Bahdanau, D. Serdyuk, K. Cho, and Y. Ben-
                  gio, “Attention-based modules for speech recognition,” in Ad-
                  vances in Neural Information Processing Systems, 2015, pp.
                  577–585.
                This is mainly used if the model wants to attend to multiple inputs parts
                at the same decoding step. We probably won't be using it since multiple sound
                frames may depend from the same character, probably not the way around.
                Note:
                    We still keep it implemented in case we want to test it. They used it in the
                    paper in the context of speech recognition, where one phoneme may depend on
                    multiple subsequent sound frames.
            name: Name to use when creating ops.
        """
        # Create normalization function
        # Setting it to None defaults in using softmax
        normalization_function = _smoothing_normalization if (smoothing == True) else None
        memory_length = memory_sequence_length if (mask_encoder == True) else None

        super(LocationSensitiveAttention, self).__init__(
            num_units=num_units,
            memory=memory,
            memory_sequence_length=memory_length,
            probability_fn=normalization_function,
            name=name)

        self.location_convolution = tf.layers.Conv1D(filters=hparams.attention_filters,
                                                     kernel_size=hparams.attention_kernel, padding='same',
                                                     use_bias=True,
                                                     bias_initializer=tf.zeros_initializer(),
                                                     name='location_features_convolution')
        self.location_layer = tf.layers.Dense(units=num_units, use_bias=False,
                                              dtype=tf.float32, name='location_features_layer')
        self._cumulate = cumulate_weights

    def __call__(self, query, state):
        """Score the query based on the keys and values.
		Args:
			query: Tensor of dtype matching `self.values` and shape
				`[batch_size, query_depth]`.
			state (previous alignments): Tensor of dtype matching `self.values` and shape
				`[batch_size, alignments_size]`
				(`alignments_size` is memory's `max_time`).
		Returns:
			alignments: Tensor of dtype matching `self.values` and shape
				`[batch_size, alignments_size]` (`alignments_size` is memory's
				`max_time`).
		"""

        previous_alignments = state
        with variable_scope.variable_scope(None, "Location_Sensitive_Attention", [query]):

            # processed_query shape [batch_size, query_depth] -> [batch_size, attention_dim]
            processed_query = self.query_layer(query) if self.query_layer else query
            # -> [batch_size, 1, attention_dim]
            processed_query = tf.expand_dims(processed_query, 1)

            # processed_location_features shape [batch_size, max_time, attention dimension]
            # [batch_size, max_time] -> [batch_size, max_time, 1]
            expanded_alignments = tf.expand_dims(previous_alignments, axis=2)
            # location features [batch_size, max_time, filters]
            f = self.location_convolution(expanded_alignments)
            # Projected location features [batch_size, max_time, attention_dim]
            processed_location_features = self.location_layer(f)

            # energy shape [batch_size, max_time]
            energy = _location_sensitive_score(processed_query, processed_location_features, self.keys)

        # alignments shape = energy shape = [batch_size, max_time]
        alignments = self._probability_fn(energy, previous_alignments)
        # Cumulate alignments
        if self._cumulate:
            next_state = alignments + previous_alignments
        else:
            next_state = alignments

        return alignments, next_state



class MultiheadAttention():
    '''Computes the multi-head attention as described in
    https://arxiv.org/abs/1706.03762.
    Args:
      num_heads: The number of attention heads.
      query: The sequence of queries. A tensor of shape :math:`[B, T_1, ...]`.
      value: The sequence to attend. A tensor of shape :math:`[B, T_2, ...]`.
        If ``None``, computes self-attention.
      num_units: The number of hidden units. If not set, it is set to the input
        dimension.
      attention_type: a string, either "dot_attention", "mlp_attention".
    Returns:
       The concatenated attention context of each head.
    '''

    def __init__(self,
                 query,
                 value,
                 num_heads=4,
                 attention_type='mlp_attention',
                 num_units=None,
                 normalize=True):
        self.query = query
        self.value = value
        self.num_heads = num_heads
        self.attention_type = attention_type
        self.num_units = num_units or query.get_shape().as_list()[-1]
        self.normalize = normalize

    def multi_head_attention(self):
        if self.num_units % self.num_heads != 0:
            raise ValueError("Multi head attention requires that num_units is a"
                             " multiple of {}".format(self.num_heads))

        with tf.variable_scope("Multihead-attention"):
            q = tf.layers.conv1d(self.query, self.num_units, 1)
            k = tf.layers.conv1d(self.value, self.num_units, 1)
            v = self.value
            qs, ks, vs = self._split_heads(q, k, v)
            if self.attention_type == 'mlp_attention':
                style_embeddings = self._mlp_attention(qs, ks, vs)
            elif self.attention_type == 'dot_attention':
                style_embeddings = self._dot_product(qs, ks, vs)
            else:
                raise ValueError('Only mlp_attention and dot_attention are supported')

            return self._combine_heads(style_embeddings)

    def _split_heads(self, q, k, v):
        '''Split the channels into multiple heads

        Returns:
             Tensors with shape [batch, num_heads, length_x, dim_x/num_heads]
        '''
        qs = tf.transpose(self._split_last_dimension(q, self.num_heads), [0, 2, 1, 3])
        ks = tf.transpose(self._split_last_dimension(k, self.num_heads), [0, 2, 1, 3])
        vs = tf.tile(tf.expand_dims(v, axis=1), [1, self.num_heads, 1, 1])
        return qs, ks, vs

    def _split_last_dimension(self, x, num_heads):
        '''Reshape x to num_heads

        Returns:
            a Tensor with shape [batch, length_x, num_heads, dim_x/num_heads]
        '''
        x_shape = shape_list(x)
        dim = x_shape[-1]
        assert dim % num_heads == 0
        return tf.reshape(x, x_shape[:-1] + [num_heads, dim // num_heads])

    def _dot_product(self, qs, ks, vs):
        '''dot-product computation

        Returns:
            a context vector with shape [batch, num_heads, length_q, dim_vs]
        '''
        qk = tf.matmul(qs, ks, transpose_b=True)
        scale_factor = (self.num_units // self.num_heads) ** -0.5
        if self.normalize:
            qk *= scale_factor
        weights = tf.nn.softmax(qk, name="dot_attention_weights")
        context = tf.matmul(weights, vs)
        return context

    def _mlp_attention(self, qs, ks, vs):
        '''MLP computation modified from https://github.com/npuichigo

        Returns:
            a context vector with shape [batch, num_heads, length_q, dim_vs]
        '''
        num_units = qs.get_shape()[-1].value
        dtype = qs.dtype

        v = tf.get_variable("attention_v", [num_units], dtype=dtype)
        if self.normalize:
            # https://github.com/tensorflow/tensorflow/blob/r1.7/tensorflow/contrib/seq2seq/python/ops/attention_wrapper.py#L470
            # Scalar used in weight normalization
            g = tf.get_variable(
                "attention_g", dtype=dtype,
                initializer=math.sqrt((1. / num_units)))
            # Bias added prior to the nonlinearity
            b = tf.get_variable(
                "attention_b", [num_units], dtype=dtype,
                initializer=tf.zeros_initializer())
            # normed_v = g * v / ||v||
            normed_v = g * v * tf.rsqrt(
                tf.reduce_sum(tf.square(v)))
            # Single layer multilayer perceptron.
            add = tf.reduce_sum(normed_v * tf.tanh(ks + qs + b), [-1], keep_dims=True)
        else:
            # Single layer multilayer perceptron.
            add = tf.reduce_sum(v * tf.tanh(ks + qs), [-1], keep_dims=True)

        # Compute attention weights.
        weights = tf.nn.softmax(tf.transpose(add, [0, 1, 3, 2]), name="mlp_attention_weights")
        # Compute attention context.
        context = tf.matmul(weights, vs)
        return context

    def _combine_heads(self, x):
        '''Combine all heads

           Returns:
               a Tensor with shape [batch, length_x, shape_x[-1] * shape_x[-3]]
        '''
        x = tf.transpose(x, [0, 2, 1, 3])
        x_shape = shape_list(x)
        return tf.reshape(x, x_shape[:-2] + [self.num_heads * x_shape[-1]])


# From https://github.com/tensorflow/tensorflow/blob/r1.7/tensorflow/contrib/seq2seq/python/ops/attention_wrapper.py
def _compute_attention(attention_mechanism, cell_output, attention_state,
                       attention_layer):
    """Computes the attention and alignments for a given attention_mechanism."""
    alignments, next_attention_state = attention_mechanism(
        cell_output, state=attention_state)

    # Reshape from [batch_size, memory_time] to [batch_size, 1, memory_time]
    expanded_alignments = array_ops.expand_dims(alignments, 1)
    # Context is the inner product of alignments and values along the 	# memory time dimension.
    # alignments shape is	#   [batch_size, 1, memory_time]
    # attention_mechanism.values shape is	#   [batch_size, memory_time, memory_size]
    # the batched matmul is over memory_time, so the output shape is	#   [batch_size, 1, memory_size].
    # we then squeeze out the singleton dim.
    context = math_ops.matmul(expanded_alignments, attention_mechanism.values)
    context = array_ops.squeeze(context, [1])

    if attention_layer is not None:
        attention = attention_layer(array_ops.concat([cell_output, context], 1))
    else:
        attention = context

    return attention, alignments, next_attention_state


def _location_sensitive_score(W_query, W_fil, W_keys):
    """Impelements Bahdanau-style (cumulative) scoring function.
	This attention is described in:
		J. K. Chorowski, D. Bahdanau, D. Serdyuk, K. Cho, and Y. Ben-
	  gio, “Attention-based modules for speech recognition,” in Ad-
	  vances in Neural Information Processing Systems, 2015, pp.
	  577–585.

	#############################################################################
			  hybrid attention (content-based + location-based)
							   f = F * α_{i-1}
	   energy = dot(v_a, tanh(W_keys(h_enc) + W_query(h_dec) + W_fil(f) + b_a))
	#############################################################################

	Args:
		W_query: Tensor, shape '[batch_size, 1, attention_dim]' to compare to location features.
		W_location: processed previous alignments into location features, shape '[batch_size, max_time, attention_dim]'
		W_keys: Tensor, shape '[batch_size, max_time, attention_dim]', typically the encoder outputs.
	Returns:
		A '[batch_size, max_time]' attention score (energy)
	"""
    # Get the number of hidden units from the trailing dimension of keys
    dtype = W_query.dtype
    num_units = W_keys.shape[-1].value or array_ops.shape(W_keys)[-1]

    v_a = tf.get_variable(
        'attention_variable', shape=[num_units], dtype=dtype,
        initializer=tf.contrib.layers.xavier_initializer())
    b_a = tf.get_variable(
        'attention_bias', shape=[num_units], dtype=dtype,
        initializer=tf.zeros_initializer())

    return tf.reduce_sum(v_a * tf.tanh(W_keys + W_query + W_fil + b_a), [2])



def _smoothing_normalization(e):
    """Applies a smoothing normalization function instead of softmax
	Introduced in:
		J. K. Chorowski, D. Bahdanau, D. Serdyuk, K. Cho, and Y. Ben-
	  gio, “Attention-based modules for speech recognition,” in Ad-
	  vances in Neural Information Processing Systems, 2015, pp.
	  577–585.

	############################################################################
						Smoothing normalization function
				a_{i, j} = sigmoid(e_{i, j}) / sum_j(sigmoid(e_{i, j}))
	############################################################################

	Args:
		e: matrix [batch_size, max_time(memory_time)]: expected to be energy (score)
			values of an attention mechanism
	Returns:
		matrix [batch_size, max_time]: [0, 1] normalized alignments with possible
			attendance to multiple memory time steps.
	"""
    return tf.nn.sigmoid(e) / tf.reduce_sum(tf.nn.sigmoid(e), axis=-1, keepdims=True)


