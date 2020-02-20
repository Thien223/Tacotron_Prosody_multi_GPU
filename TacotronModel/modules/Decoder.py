from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import tensorflow as tf
from tensorflow.contrib.rnn import RNNCell
from tensorflow.contrib.seq2seq.python.ops import decoder
from tensorflow.contrib.seq2seq.python.ops import helper as helper_py
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.layers import base as layers_base
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.util import nest

from TacotronModel.modules.Attention import _compute_attention
from TacotronModel.modules.Encoder import ZoneoutLSTMCell
from Utils.Hyperparams import hparams

_zero_state_tensors = rnn_cell_impl._zero_state_tensors


class CustomDecoderOutput(
    collections.namedtuple("CustomDecoderOutput", ("rnn_output", "token_output", "sample_id"))):
    pass


class CustomDecoder(decoder.Decoder):
    """Custom sampling decoder.

	Allows for stop token prediction at inference time
	and returns equivalent loss in training time.

	Note:
	Only use this decoder with Tacotron 2 as it only accepts tacotron custom helpers
	"""

    def __init__(self, cell, helper, initial_state, output_layer=None):
        """Initialize CustomDecoder.
		Args:
			cell: An `RNNCell` instance.
			helper: A `Helper` instance.
			initial_state: A (possibly nested tuple of...) tensors and TensorArrays.
				The initial state of the RNNCell.
			output_layer: (Optional) An instance of `tf.layers.Layer`, i.e.,
				`tf.layers.Dense`. Optional layer to apply to the RNN output prior
				to storing the result or sampling.
		Raises:
			TypeError: if `cell`, `helper` or `output_layer` have an incorrect type.
		"""
        rnn_cell_impl.assert_like_rnncell(type(cell), cell)
        if not isinstance(helper, helper_py.Helper):
            raise TypeError("helper must be a Helper, received: %s" % type(helper))
        if (output_layer is not None
                and not isinstance(output_layer, layers_base.Layer)):
            raise TypeError(
                "output_layer must be a Layer, received: %s" % type(output_layer))
        self._cell = cell
        self._helper = helper
        self._initial_state = initial_state
        self._output_layer = output_layer

    @property
    def batch_size(self):
        return self._helper.batch_size

    def _rnn_output_size(self):
        size = self._cell.output_size
        if self._output_layer is None:
            return size
        else:
            # To use layer's compute_output_shape, we need to convert the
            # RNNCell's output_size entries into shapes with an unknown
            # batch size.  We then pass this through the layer's
            # compute_output_shape and read off all but the first (batch)
            # dimensions to get the output size of the rnn with the layer
            # applied to the top.
            output_shape_with_unknown_batch = nest.map_structure(
                lambda s: tensor_shape.TensorShape([None]).concatenate(s),
                size)
            layer_output_shape = self._output_layer._compute_output_shape(  # pylint: disable=protected-access
                output_shape_with_unknown_batch)
            return nest.map_structure(lambda s: s[1:], layer_output_shape)

    @property
    def output_size(self):
        # Return the cell output and the id
        return CustomDecoderOutput(
            rnn_output=self._rnn_output_size(),
            token_output=self._helper.token_output_size,
            sample_id=self._helper.sample_ids_shape)

    @property
    def output_dtype(self):
        # Assume the dtype of the cell is the output_size structure
        # containing the input_state's first component's dtype.
        # Return that structure and the sample_ids_dtype from the helper.
        dtype = nest.flatten(self._initial_state)[0].dtype
        return CustomDecoderOutput(
            nest.map_structure(lambda _: dtype, self._rnn_output_size()),
            tf.float32,
            self._helper.sample_ids_dtype)

    def initialize(self, name=None):
        """Initialize the decoder.
        Args:
            name: Name scope for any created operations.
        Returns:
            `(finished, first_inputs, initial_state)`.
        """
        return self._helper.initialize() + (self._initial_state,)

    def step(self, time, inputs, state, name=None):
        """Perform a custom decoding step.
        Enables for dyanmic <stop_token> prediction
        Args:
            time: scalar `int32` tensor.
            inputs: A (structure of) input tensors.
            state: A (structure of) state tensors and TensorArrays.
            name: Name scope for any created operations.
        Returns:
            `(outputs, next_state, next_inputs, finished)`.
        """
        with ops.name_scope(name, "CustomDecoderStep", (time, inputs, state)):
            # Call outputprojection wrapper cell
            (cell_outputs, stop_token), cell_state = self._cell(inputs, state)

            # apply output_layer (if existant)
            if self._output_layer is not None:
                cell_outputs = self._output_layer(cell_outputs)
            sample_ids = self._helper.sample(
                time=time, outputs=cell_outputs, state=cell_state)

            (finished, next_inputs, next_state) = self._helper.next_inputs(
                time=time,
                outputs=cell_outputs,
                state=cell_state,
                sample_ids=sample_ids,
                stop_token_prediction=stop_token)

        outputs = CustomDecoderOutput(cell_outputs, stop_token, sample_ids)
        return (outputs, next_state, next_inputs, finished)


class TacotronDecoderCell(RNNCell):
    """Tactron 2 Decoder Cell
	Decodes encoder output and previous mel frames into next r frames

	Decoder Step i:
		1) Prenet to compress last output information
		2) Concat compressed inputs with previous context vector (input feeding) *
		3) Decoder RNN (actual decoding) to predict current state s_{i} *
		4) Compute new context vector c_{i} based on s_{i} and a cumulative sum of previous alignments *
		5) Predict new output y_{i} using s_{i} and c_{i} (concatenated)
		6) Predict <stop_token> output ys_{i} using s_{i} and c_{i} (concatenated)

	* : This is typically taking a vanilla LSTM, wrapping it using tensorflow's attention wrapper,
	and wrap that with the prenet before doing an input feeding, and with the prediction layer
	that uses RNN states to project on output space. Actions marked with (*) can be replaced with
	tensorflow's attention wrapper call if it was using cumulative alignments instead of previous alignments only.
	"""

    def __init__(self, prenet, attention_mechanism, rnn_cell, frame_projection, stop_projection):
        """Initialize decoder parameters

		Args:
		    prenet: A tensorflow fully connected layer acting as the decoder pre-net
		    attention_mechanism: A _BaseAttentionMechanism instance, usefull to
			    learn encoder-decoder alignments
		    rnn_cell: Instance of RNNCell, main body of the decoder
		    frame_projection: tensorflow fully connected layer with r * num_mels output units
		    stop_projection: tensorflow fully connected layer, expected to project to a scalar
			    and through a sigmoid activation
			mask_finished: Boolean, Whether to mask decoder frames after the <stop_token>
		"""
        super(TacotronDecoderCell, self).__init__()
        # Initialize decoder layers
        self._prenet = prenet
        self._attention_mechanism = attention_mechanism
        self._cell = rnn_cell
        self._frame_projection = frame_projection
        self._stop_projection = stop_projection
        self._attention_layer_size = self._attention_mechanism.values.get_shape()[-1].value

    def _batch_size_checks(self, batch_size, error_message):
        return [check_ops.assert_equal(batch_size,
                                       self._attention_mechanism.batch_size,
                                       message=error_message)]

    @property
    def output_size(self):
        return self._frame_projection.shape

    @property
    def state_size(self):
        """The `state_size` property of `TacotronDecoderCell`.

		Returns:
		  An `TacotronDecoderCell` tuple containing shapes used by this object.
		"""
        return TacotronDecoderCellState(
            cell_state=self._cell._cell.state_size,
            time=tensor_shape.TensorShape([]),
            attention=self._attention_layer_size,
            alignments=self._attention_mechanism.alignments_size,
            alignment_history=())

    def zero_state(self, batch_size, dtype):
        """Return an initial (zero) state tuple for this `AttentionWrapper`.

		Args:
		  batch_size: `0D` integer tensor: the batch size.
		  dtype: The internal state data type.
		Returns:
		  An `TacotronDecoderCellState` tuple containing zeroed out tensors and,
		  possibly, empty `TensorArray` objects.
		Raises:
		  ValueError: (or, possibly at runtime, InvalidArgument), if
			`batch_size` does not match the output size of the encoder passed
			to the wrapper object at initialization time.
		"""
        with ops.name_scope(type(self).__name__ + "ZeroState", values=[batch_size]):
            cell_state = self._cell._cell.zero_state(batch_size, dtype)
            error_message = (
                    "When calling zero_state of TacotronDecoderCell %s: " % self._base_name +
                    "Non-matching batch sizes between the memory "
                    "(encoder output) and the requested batch size.")
            with ops.control_dependencies(
                    self._batch_size_checks(batch_size, error_message)):
                cell_state = nest.map_structure(
                    lambda s: array_ops.identity(s, name="checked_cell_state"),
                    cell_state)
            return TacotronDecoderCellState(
                cell_state=cell_state,
                time=array_ops.zeros([], dtype=tf.int32),
                attention=_zero_state_tensors(self._attention_layer_size, batch_size,
                                              dtype),
                alignments=self._attention_mechanism.initial_alignments(batch_size, dtype),
                alignment_history=tensor_array_ops.TensorArray(dtype=dtype, size=0,
                                                               dynamic_size=True))

    def __call__(self, inputs, state):
        # Information bottleneck (essential for learning attention)
        ## prenet output
        prenet_output = self._prenet(inputs)

        # Concat context vector and prenet output to form LSTM cells input (input feeding)
        # put prenet output and state
        LSTM_input = tf.concat([prenet_output, state.attention], axis=-1)

        # Unidirectional LSTM layers
        LSTM_output, next_cell_state = self._cell(LSTM_input, state.cell_state)
        print('LSTM_output.shape {}'.format(LSTM_output.shape))

        # Compute the attention (context) vector and alignments using
        # the new decoder cell hidden state as query vector
        # and cumulative alignments to extract location features
        # The choice of the new cell hidden state (s_{i}) of the last
        # decoder RNN Cell is based on Luong et Al. (2015):
        # https://arxiv.org/pdf/1508.04025.pdf
        previous_alignments = state.alignments
        previous_alignment_history = state.alignment_history
        context_vector, alignments, cumulated_alignments = _compute_attention(self._attention_mechanism,
                                                                              LSTM_output,
                                                                              previous_alignments,
                                                                              attention_layer=None)

        # Concat LSTM outputs and context vector to form projections inputs
        projections_input = tf.concat([LSTM_output, context_vector], axis=-1)
        # Compute predicted frames and predicted <stop_token>
        cell_outputs = self._frame_projection(projections_input)
        print('projections_input.shape {}'.format(projections_input.shape))
        print('cell_outputs.shape {}'.format(cell_outputs.shape))

        stop_tokens = self._stop_projection(projections_input)

        # Save alignment history
        alignment_history = previous_alignment_history.write(state.time, alignments)

        # Prepare next decoder state
        next_state = TacotronDecoderCellState(
            time=state.time + 1,
            cell_state=next_cell_state,
            attention=context_vector,
            alignments=cumulated_alignments,
            alignment_history=alignment_history)

        return (cell_outputs, stop_tokens), next_state


class DecoderRNN():
    '''2 unidirection LSTM layers
    :arg
        is_training: whether mode is used for training
        layers: number of LSTM layers
        size: hidden unit size
        zoneout: zoneout ratio
        scope: model scope
    '''

    def __init__(self, is_training, layers=2, size=256, zoneout=0.1, scope='decoder_rnn'):
        super(DecoderRNN, self).__init__()
        self.is_training = is_training
        self.layers = layers
        self.size = size
        self.zoneout = zoneout
        self.scope = scope
        # create  LSTM layers
        self.rnn_layers = [ZoneoutLSTMCell(size, is_training,
                                           zoneout_factor_cell=zoneout,
                                           zoneout_factor_output=zoneout,
                                           name='decoder_LSTM_{}'.format(i + 1)) for i in range(layers)]
        self._cell = tf.contrib.rnn.MultiRNNCell(self.rnn_layers, state_is_tuple=True)

    def __call__(self, input, states):
        with tf.variable_scope(self.scope):
            return self._cell(input, states)


class Prenet():
    """Two fully connected layers used as an information bottleneck for the attention.
		Args:
			layers_sizes: list of integers, the length of the list (2) represents the number of pre-net
				layers and the list values (256) represent the layers number of units
			activation: callable, activation functions of the prenet layers.
			scope: Prenet scope.
		"""

    def __init__(self, is_training, layers_sizes, drop_rate=0.5, activation=tf.nn.relu, scope=None):
        super(Prenet, self).__init__()
        self.drop_rate = drop_rate
        self.layers_sizes = layers_sizes
        self.activation = activation
        self.is_training = is_training

        self.scope = 'prenet' if scope is None else scope

    def __call__(self, inputs):
        x = inputs
        with tf.variable_scope(self.scope):
            for i, size in enumerate(self.layers_sizes):
                dense = tf.layers.dense(x, units=size, activation=self.activation,
                                        name='dense_{}'.format(i + 1))
                # The paper discussed introducing diversity in generation at inference time
                # by using a dropout of 0.5 only in prenet layers (in both training and inference).
                x = tf.layers.dropout(dense, rate=self.drop_rate, training=True,
                                      name='dropout_{}'.format(i + 1) + self.scope)
        return x


class FrameProjection():
    """Projection layer to r * num_mels dimensions or num_mels dimensions
	"""

    def __init__(self, shape=hparams.num_mels, activation=None, scope=None):
        """
		Args:
			shape: integer, dimensionality of output space (r*n_mels for decoder or n_mels for postnet)
			activation: callable, activation function
			scope: FrameProjection scope.
		"""
        super(FrameProjection, self).__init__()

        self.shape = shape
        self.activation = activation
        self.scope = 'Linear_projection' if scope is None else scope
        self.dense = tf.layers.Dense(units=shape, activation=activation, name='projection_{}'.format(self.scope))

    def __call__(self, inputs):
        with tf.variable_scope(self.scope):
            # If activation==None, this returns a simple Linear projection
            # else the projection will be passed through an activation function
            # output = tf.layers.dense(inputs, units=self.shape, activation=self.activation,
            # 	name='projection_{}'.format(self.scope))
            output = self.dense(inputs)
            return output


class StopProjection():
    """Projection to a scalar and through a sigmoid activation
	"""

    def __init__(self, is_training, shape=1, activation=tf.nn.sigmoid, scope=None):
        """
		Args:
			is_training: Boolean, to control the use of sigmoid function as it is useless to use it
				during training since it is integrate inside the sigmoid_crossentropy loss
			shape: integer, dimensionality of output space. Defaults to 1 (scalar)
			activation: callable, activation function. only used during inference
			scope: StopProjection scope.
		"""
        super(StopProjection, self).__init__()
        self.is_training = is_training
        self.shape = shape
        self.activation = activation
        self.scope = 'stop_token_projection' if scope is None else scope

    def __call__(self, inputs):
        with tf.variable_scope(self.scope):
            output = tf.layers.dense(inputs, units=self.shape,
                                     activation=None, name='projection_{}'.format(self.scope))

            # During training, don't use activation as it is integrated inside the sigmoid_cross_entropy loss function
            if self.is_training:
                return output
            return self.activation(output)


###############################################################################
### Attention definition######
###############################################################################



class TacotronDecoderCellState(
    collections.namedtuple("TacotronDecoderCellState",
                           ("cell_state", "attention", "time", "alignments",
                            "alignment_history"))):
    """`namedtuple` storing the state of a `TacotronDecoderCell`.
	Contains:
	  - `cell_state`: The state of the wrapped `RNNCell` at the previous time
		step.
	  - `attention`: The attention emitted at the previous time step.
	  - `time`: int32 scalar containing the current time step.
	  - `alignments`: A single or tuple of `Tensor`(s) containing the alignments
		 emitted at the previous time step for each attention mechanism.
	  - `alignment_history`: a single or tuple of `TensorArray`(s)
		 containing alignment matrices from all time steps for each attention
		 mechanism. Call `stack()` on each to convert to a `Tensor`.
	"""

    def replace(self, **kwargs):
        """Clones the current state while overwriting components provided by kwargs."""
        return super(TacotronDecoderCellState, self)._replace(**kwargs)
