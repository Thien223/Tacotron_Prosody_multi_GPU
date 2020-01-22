from TacotronModel.modules import Encoder, Decoder, Postnet, Attention
import tensorflow as tf
from Utils.Helpers import TacoTestHelper, TacoTrainingHelper
from tensorflow.contrib.seq2seq import dynamic_decode
from Utils.Infolog import log
from Utils.Utils import MaskedMSE, MaskedSigmoidCrossEntropy, shape_list, split_func
from Utils.TextProcessing.HangulUtils import hangul_symbol_1, hangul_symbol_2, hangul_symbol_3, hangul_symbol_4, \
    hangul_symbol_5
from tensorflow.contrib.rnn import GRUCell

def reference_encoder(inputs, filters, kernel_size, strides, encoder_cell, is_training, scope='ref_encoder'):
    with tf.variable_scope(scope):
        ref_outputs = tf.expand_dims(inputs,axis=-1)
        # CNN stack
        for i, channel in enumerate(filters):
            ref_outputs = Encoder.conv2d(ref_outputs, channel, kernel_size, strides, tf.nn.relu, is_training, 'conv2d_%d' % i)

        shapes = shape_list(ref_outputs)
        ref_outputs = tf.reshape(
            ref_outputs,
            shapes[:-2] + [shapes[2] * shapes[3]])
        # RNN
        encoder_outputs, encoder_state = tf.nn.dynamic_rnn(
            encoder_cell,
            ref_outputs,
            dtype=tf.float32)
        reference_state = tf.layers.dense(encoder_outputs[:,-1,:], 128, activation=tf.nn.tanh) # [N, 128]
        return reference_state


class Tacotron():
    '''Tacotron model, the wrapper of Encoder and Decoder model
    :arg
        hparams: hyper parameters to use'''

    def __init__(self, hparams):
        self.hparams = hparams
        self.tower_decoder_output = []
        self.tower_alignments = []
        self.tower_stop_token_prediction = []
        self.tower_mel_outputs = []
        self.tower_linear_outputs = []
        self.tower_ref_output = []

    def initialize(self, inputs, input_lengths, mel_targets=None, stop_token_targets=None, linear_targets=None,
                   targets_lengths=None, GTA=False, global_step=None, is_training=False,
                   is_evaluating=False, split_infos=None, reference_mel=None):
        """
                Initializes the model for inference

                sets "mel_outputs" and "alignments" fields.

                Args:
                    - inputs: int32 Tensor with shape [N, T_in] where N is batch size, T_in is number of
                      steps in the input time series, and values are character IDs
                    - input_lengths: int32 Tensor with shape [N] where N is batch size and values are the lengths
                    of each sequence in inputs.
                    - mel_targets: float32 Tensor with shape [N, T_out, M] where N is batch size, T_out is number
                    of steps in the output time series, M is num_mels, and values are entries in the mel
                    spectrogram. Only needed for training.
                """

        ### checking for conditions
        if mel_targets is None and stop_token_targets is not None:
            raise ValueError('no mel targets were provided but token_targets were given')
        if mel_targets is not None and stop_token_targets is None and not GTA:
            raise ValueError('Mel targets are provided without corresponding token_targets')
        if not GTA and self.hparams.predict_linear == True and linear_targets is None and is_training:
            raise ValueError(
                'Model is set to use post processing to predict linear spectrograms in training but no linear targets given!')
        if GTA and linear_targets is not None:
            raise ValueError('Linear spectrogram prediction is not supported in GTA mode!')
        if is_training and self.hparams.mask_decoder and targets_lengths is None:
            raise RuntimeError('Model set to mask paddings but no targets lengths provided for the mask!')
        if is_training and is_evaluating:
            raise RuntimeError('Model can not be in training and evaluation modes at the same time!')

        ### get symbol to create embedding lookup table
        if self.hparams.hangul_type == 1:
            hangul_symbol = hangul_symbol_1
        elif self.hparams.hangul_type == 2:
            hangul_symbol = hangul_symbol_2
        elif self.hparams.hangul_type == 3:
            hangul_symbol = hangul_symbol_3
        elif self.hparams.hangul_type == 4:
            hangul_symbol = hangul_symbol_4
        else:
            hangul_symbol = hangul_symbol_5


        split_device = '/cpu:0'
        with tf.device(split_device):
            hp = self.hparams
            lout_int = [tf.int32] * hp.tacotron_num_gpus
            lout_float = [tf.float32] * hp.tacotron_num_gpus

            tower_input_lengths = tf.split(input_lengths, num_or_size_splits=hp.tacotron_num_gpus, axis=0)
            tower_targets_lengths = tf.split(targets_lengths, num_or_size_splits=hp.tacotron_num_gpus,axis=0) if targets_lengths is not None else targets_lengths


            p_inputs = tf.py_func(split_func, [inputs, split_infos[:, 0]], lout_int)
            p_mel_targets = tf.py_func(split_func, [mel_targets, split_infos[:, 1]],
                                       lout_float) if mel_targets is not None else mel_targets
            p_stop_token_targets = tf.py_func(split_func, [stop_token_targets, split_infos[:, 2]],
                                              lout_float) if stop_token_targets is not None else stop_token_targets
            p_linear_targets = tf.py_func(split_func, [linear_targets, split_infos[:, 3]],
                                          lout_float) if linear_targets is not None else linear_targets

            tower_inputs = []
            tower_mel_targets = []
            tower_stop_token_targets = []
            tower_linear_targets = []
            ##todo:
            tower_ref_audio = []

            batch_size = tf.shape(inputs)[0]
            mel_channels = hp.num_mels
            linear_channels = hp.num_freq
            for i in range(hp.tacotron_num_gpus):
                tower_inputs.append(tf.reshape(p_inputs[i], [batch_size, -1]))
                if p_mel_targets is not None:
                    tower_mel_targets.append(tf.reshape(p_mel_targets[i], [batch_size, -1, mel_channels]))
                if p_stop_token_targets is not None:
                    tower_stop_token_targets.append(tf.reshape(p_stop_token_targets[i], [batch_size, -1]))
                if p_linear_targets is not None:
                    tower_linear_targets.append(tf.reshape(p_linear_targets[i], [batch_size, -1, linear_channels]))
                if is_training:
                    ## if training, add mel_targets as ref_audio
                    tower_ref_audio.append(tf.reshape(p_mel_targets[i], [batch_size, -1, mel_channels]))

        T2_output_range = (-hp.max_abs_value, hp.max_abs_value) if hp.symmetric_mels else (0, hp.max_abs_value)
        tower_embedded_inputs = []
        tower_enc_conv_output_shape = []
        tower_encoder_outputs = []
        tower_residual = []
        tower_projected_residual = []
        # tower_ref_encoder = []

        # 1. Declare GPU Devices
        gpus = ["/gpu:{}".format(i) for i in range(hp.tacotron_num_gpus)]
        for i in range(hp.tacotron_num_gpus):
            with tf.device(tf.train.replica_device_setter(ps_tasks=1, ps_device="/cpu:0", worker_device=gpus[i])):
                with tf.variable_scope('inference') as scope:
                    assert hp.tacotron_teacher_forcing_mode in ('constant', 'scheduled')
                    if hp.tacotron_teacher_forcing_mode == 'scheduled' and is_training:
                        assert global_step is not None

                    ## reference embedding:
                    ##if there are input reference audio
                    # if tower_ref_audio[i] is not None:
                    # 	print(tower_ref_audio[i])
                    # 	ref_encoder = reference_encoder(inputs=tower_ref_audio[i], is_training=is_training)
                    # 	self.tower_ref_output.append(ref_encoder)
                    # 	log('Tacotron.py line 180 ref_encoder.shape: {}'.format(ref_encoder.shape))
                    # 	ref_attention = Attention.MultiheadsAttention(
                    # 		query=tf.expand_dims(ref_encoder, axis=1),
                    # 		value=tf.tanh(tf.tile(tf.expand_dims(self.style_tokens, axis=0), [batch_size,1,1])),
                    # 		attention_heads=self.hparams.attention_heads,
                    # 		num_units=128,
                    # 		normalize=True
                    # 	)
                    # 	style_embedding = ref_attention.multi_heads_attention()
                    # 	log('Tacotron.py line 188 style_embedding.shape: {}'.format(style_embedding.shape))
                    # else: ### if there is not input reference audio, use random
                    # 	rand_weight = tf.nn.softmax(tf.random_uniform([hp.attention_heads, hp.num_tokens], maxval=1.0, dtype=tf.float32),name='random_weight_gst')
                    # 	style_embedding = tf.reshape(tf.matmul(rand_weight,tf.nn.tanh(self.style_tokens), [1,1]+[hp.attention_heads+self.style_tokens.get_shape().as_list()[1]]))
                    #
                    # 	log('Tacotron.py line 193 style_embedding.shape: {}'.format(style_embedding.shape))

                    if hp.use_gst:
                        # Global style tokens (GST)
                        gst_tokens = tf.get_variable(
                            'style_tokens', [hp.num_gst, hp.style_embed_depth // hp.num_heads], dtype=tf.float32,
                            initializer=tf.truncated_normal_initializer(stddev=0.5))
                        self.gst_tokens = gst_tokens

                    # Embeddings ==> [batch_size, sequence_length, embedding_dim]
                    self.embedding_table = tf.get_variable(
                        'inputs_embedding', [len(hangul_symbol), hp.embedding_dim], dtype=tf.float32)
                    embedded_inputs = tf.nn.embedding_lookup(self.embedding_table, tower_inputs[i])

                    self.embedded_inputs_ = embedded_inputs
                    # Encoder Cell ==> [batch_size, encoder_steps, encoder_lstm_units]
                    encoder_cell = Encoder.EncoderCell(
                        Encoder.EncoderConvolution(is_training, hparams=hp, scope='encoder_convolutions'),
                        Encoder.EncoderLSTM(is_training, size=hp.enc_lstm_hidden_size,
                                            zoneout=hp.zoneout_rate, scope='encoder_LSTM'))
                    encoder_outputs = encoder_cell(embedded_inputs, tower_input_lengths[i])
                    # # ### append reference embedding to encoder output
                    if is_training:
                        ### on training,   reference_mel is None, set it to target mel
                        reference_mel = mel_targets

                    if reference_mel is not None:
                        # Reference encoder
                        refnet_outputs = reference_encoder(
                            reference_mel,
                            filters=hp.reference_filters,
                            kernel_size=(3, 3),
                            strides=(2, 2),
                            encoder_cell=GRUCell(hp.reference_depth),
                            is_training=is_training)  # [N, 128]
                        self.refnet_outputs = refnet_outputs
                        if hp.use_gst:
                            style_attention = Attention.MultiheadAttention(
                                query=tf.expand_dims(refnet_outputs, axis=1),  # [N, 1, 128]
                                value=tf.tanh(tf.tile(tf.expand_dims(gst_tokens, axis=0), [batch_size, 1, 1])),
                                # [N, hp.num_gst, 256/hp.num_heads]
                                num_heads=hp.num_heads,
                                num_units=hp.style_att_dim,
                                attention_type=hp.style_att_type)
                            style_embeddings = style_attention.multi_head_attention()  # [N, 1, 256]
                        else:
                            style_embeddings = tf.expand_dims(refnet_outputs, axis=1)  # [N, 1, 128]
                        # Add style embedding to every text encoder state
                        style_embeddings = tf.tile(style_embeddings, [1, shape_list(encoder_outputs)[1], 1])  # [N, T_in, 128]


                    else:
                        print("Use random weight for GST.")
                        random_weights = tf.random_uniform([hp.num_heads, hp.num_gst], maxval=1.0, dtype=tf.float32)
                        random_weights = tf.nn.softmax(random_weights, name="random_weights")
                        style_embeddings = tf.matmul(random_weights, tf.nn.tanh(gst_tokens))
                        style_embeddings = tf.reshape(style_embeddings, [1, 1] + [hp.num_heads * gst_tokens.get_shape().as_list()[1]])
                        # Add style embedding to every text encoder state
                        style_embeddings = tf.tile(style_embeddings, [shape_list(encoder_outputs)[0], shape_list(encoder_outputs)[1], 1])  # [N, T_in, 128]


                    encoder_outputs = tf.concat([encoder_outputs, style_embeddings], axis=-1)

                    self.encoder_outputs = encoder_outputs
                    print('encoder_outputs.shape after {}'.format(encoder_outputs.shape))

                    # For shape visualization purpose
                    enc_conv_output_shape = self.encoder_outputs.shape

                    # Decoder Parts
                    # Attention Decoder Prenet
                    prenet = Decoder.Prenet(is_training, layers_sizes=hp.prenet_layers, drop_rate=hp.dropout_rate,
                                            scope='decoder_prenet')
                    # attention_mechanism = BahdanauMonotonicAttention(num_units=hp.attention_dim, memory=encoder_outputs,
                    # 												 memory_sequence_length=tf.reshape(
                    # 													 tower_input_lengths[i], [-1]), normalize=True)
                    # Attention Mechanism (original)

                    print('memory.shape {}'.format(encoder_outputs.shape))
                    attention_mechanism = Attention.LocationSensitiveAttention(
                        num_units = hp.attention_dim,
                        memory=encoder_outputs,
                        hparams=hp,
                        mask_encoder=hp.mask_encoder,
                        memory_sequence_length=tf.reshape(tower_input_lengths[i], [-1]),
                        smoothing=hp.smoothing,
                        cumulate_weights=hp.cumulative_weights)
                    # Decoder LSTM Cells
                    decoder_lstm = Decoder.DecoderRNN(is_training=is_training, layers=2,
                                                      size=hp.decoder_lstm_units, zoneout=hp.zoneout_rate,
                                                      scope='decoder_LSTM')
                    # Frames Projection layer
                    frame_projection = Decoder.FrameProjection(shape=hp.num_mels * hp.outputs_per_step,
                                                               scope='linear_transform_projection')
                    # <stop_token> projection layer
                    stop_projection = Decoder.StopProjection(is_training, shape=hp.outputs_per_step,
                                                             scope='stop_token_projection')

                    # Decoder Cell ==> [batch_size, decoder_steps, num_mels * r] (after decoding)
                    decoder_cell = Decoder.TacotronDecoderCell(
                        prenet,
                        attention_mechanism,
                        decoder_lstm,
                        frame_projection,
                        stop_projection)
                    # Define the helper for our decoder
                    if is_training or is_evaluating or GTA:
                        self.helper = TacoTrainingHelper(batch_size, tower_mel_targets[i], hp, GTA, is_evaluating,
                                                         global_step)
                    else:
                        self.helper = TacoTestHelper(batch_size, hp)

                    # initial decoder state
                    decoder_init_state = decoder_cell.zero_state(batch_size=batch_size, dtype=tf.float32)

                    # Only use max iterations at synthesis time
                    max_iters = hp.max_iters if not (is_training or is_evaluating) else None


                    # Decode
                    (frames_prediction, stop_token_prediction, _), final_decoder_state, _ = dynamic_decode(Decoder.CustomDecoder(decoder_cell, self.helper, decoder_init_state), maximum_iterations=max_iters,
                                                                                                           swap_memory=hp.tacotron_swap_with_cpu)
                    # Reshape outputs to be one output per entry
                    # ==> [batch_size, non_reduced_decoder_steps (decoder_steps * r), num_mels]
                    decoder_output = tf.reshape(frames_prediction, [batch_size, -1, hp.num_mels])
                    stop_token_prediction = tf.reshape(stop_token_prediction, [batch_size, -1])
                    if hp.clip_outputs:
                        decoder_output = tf.minimum(
                            tf.maximum(decoder_output, T2_output_range[0] - hp.lower_bound_decay), T2_output_range[1])

                    # Postnet
                    postnet = Postnet.Postnet(is_training, hparams=hp, scope='postnet_convolutions')

                    # Compute residual using post-net ==> [batch_size, decoder_steps * r, postnet_channels]
                    print('decoder_output.shape {}'.format(decoder_output.shape))

                    residual = postnet(decoder_output)
                    print('residual.shape {}'.format(residual.shape))

                    # Project residual to same dimension as mel spectrogram
                    # ==> [batch_size, decoder_steps * r, num_mels]
                    residual_projection = Decoder.FrameProjection(hp.num_mels, scope='postnet_projection')

                    projected_residual = residual_projection(residual)

                    print('projected_residual.shape {}'.format(projected_residual.shape))

                    # Compute the mel spectrogram
                    mel_outputs = decoder_output + projected_residual
                    print('mel_outputs.shape {}'.format(mel_outputs.shape))

                    if hp.clip_outputs:
                        mel_outputs = tf.minimum(tf.maximum(mel_outputs, T2_output_range[0] - hp.lower_bound_decay),
                                                 T2_output_range[1])

                    # Grab alignments from the final decoder state
                    alignments = tf.transpose(final_decoder_state.alignment_history.stack(), [1, 2, 0])

                    self.tower_decoder_output.append(decoder_output)
                    self.tower_alignments.append(alignments)
                    self.tower_stop_token_prediction.append(stop_token_prediction)
                    self.tower_mel_outputs.append(mel_outputs)
                    tower_embedded_inputs.append(embedded_inputs)
                    tower_enc_conv_output_shape.append(enc_conv_output_shape)
                    tower_encoder_outputs.append(encoder_outputs)
                    tower_residual.append(residual)
                    tower_projected_residual.append(projected_residual)
            log('initialization done {}'.format(gpus[i]))

        if is_training:
            self.ratio = self.helper._ratio
        self.tower_inputs = tower_inputs
        self.tower_input_lengths = tower_input_lengths
        self.tower_mel_targets = tower_mel_targets
        self.tower_linear_targets = tower_linear_targets
        self.tower_targets_lengths = tower_targets_lengths
        self.tower_stop_token_targets = tower_stop_token_targets
        self.all_vars = tf.trainable_variables()

        log('Initialized Tacotron model. Dimensions (? = dynamic shape): ')
        log('  Train mode:               {}'.format(is_training))
        log('  Eval mode:                {}'.format(is_evaluating))
        log('  Synthesis mode:           {}'.format(not (is_training or is_evaluating)))
        log('  Input:                    {}'.format(inputs.shape))
        for i in range(hp.tacotron_num_gpus):
            log('  device:                   {}'.format(i))
            log('  embedding:                {}'.format(tower_embedded_inputs[i].shape))
            log('  enc conv out:             {}'.format(tower_enc_conv_output_shape[i]))
            log('  encoder out:              {}'.format(tower_encoder_outputs[i].shape))
            log('  decoder out:              {}'.format(self.tower_decoder_output[i].shape))
            log('  residual out:             {}'.format(tower_residual[i].shape))
            log('  projected residual out:   {}'.format(tower_projected_residual[i].shape))
            log('  mel out:                  {}'.format(self.tower_mel_outputs[i].shape))
            log('  <stop_token> out:         {}'.format(self.tower_stop_token_prediction[i].shape))

    def add_loss(self):
        '''Adds loss to the model. Sets "loss" field. initialize must have been called.'''
        hp = self.hparams
        self.tower_before_loss = []
        self.tower_after_loss = []
        self.tower_stop_token_loss = []
        self.tower_regularization_loss = []
        self.tower_linear_loss = []
        self.tower_loss = []

        total_before_loss = 0
        total_after_loss = 0
        total_stop_token_loss = 0
        total_regularization_loss = 0
        total_linear_loss = 0
        total_loss = 0

        gpus = ["/gpu:{}".format(i) for i in range(hp.tacotron_num_gpus)]

        for i in range(hp.tacotron_num_gpus):
            with tf.device(tf.train.replica_device_setter(ps_tasks=1, ps_device="/cpu:0", worker_device=gpus[i])):
                with tf.variable_scope('loss') as scope:
                    if hp.mask_decoder:
                        # Compute loss of predictions before postnet
                        before = MaskedMSE(self.tower_mel_targets[i], self.tower_decoder_output[i],
                                           self.tower_targets_lengths[i],
                                           hparams=self.hparams)
                        # Compute loss after postnet
                        after = MaskedMSE(self.tower_mel_targets[i], self.tower_mel_outputs[i],
                                          self.tower_targets_lengths[i],
                                          hparams=self.hparams)
                        # Compute <stop_token> loss (for learning dynamic generation stop)
                        stop_token_loss = MaskedSigmoidCrossEntropy(self.tower_stop_token_targets[i],
                                                                    self.tower_stop_token_prediction[i],
                                                                    self.tower_targets_lengths[i],
                                                                    hparams=self.hparams)
                        linear_loss = 0.
                    else:
                        # Compute loss of predictions before postnet
                        before = tf.losses.mean_squared_error(self.tower_mel_targets[i], self.tower_decoder_output[i])
                        # Compute loss after postnet
                        after = tf.losses.mean_squared_error(self.tower_mel_targets[i], self.tower_mel_outputs[i])
                        # Compute <stop_token> loss (for learning dynamic generation stop)
                        stop_token_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                            labels=self.tower_stop_token_targets[i],
                            logits=self.tower_stop_token_prediction[i]))

                        if hp.predict_linear:
                            # Compute linear loss
                            # From https://github.com/keithito/tacotron/blob/tacotron2-work-in-progress/models/tacotron.py
                            # Prioritize loss for frequencies under 2000 Hz.
                            l1 = tf.abs(self.tower_linear_targets[i] - self.tower_linear_outputs[i])
                            n_priority_freq = int(2000 / (hp.sample_rate * 0.5) * hp.num_freq)
                            linear_loss = 0.5 * tf.reduce_mean(l1) + 0.5 * tf.reduce_mean(l1[:, :, 0:n_priority_freq])
                        else:
                            linear_loss = 0.

                    # Compute the regularization weight
                    if hp.tacotron_scale_regularization:
                        reg_weight_scaler = 1. / (2 * hp.max_abs_value) if hp.symmetric_mels else 1. / (
                            hp.max_abs_value)
                        reg_weight = hp.tacotron_reg_weight * reg_weight_scaler
                    else:
                        reg_weight = hp.tacotron_reg_weight

                    # Regularize variables
                    # Exclude all types of bias, RNN (Bengio et al. On the difficulty of training recurrent neural networks), embeddings and prediction projection layers.
                    # Note that we consider attention mechanism v_a weights as a prediction projection layer and we don't regularize it. (This gave better stability)
                    regularization = tf.add_n([tf.nn.l2_loss(v) for v in self.all_vars
                                               if not (
                                'bias' in v.name or 'Bias' in v.name or '_projection' in v.name or 'inputs_embedding' in v.name
                                or 'RNN' in v.name or 'LSTM' in v.name)]) * reg_weight

                    # Compute final loss term
                    self.tower_before_loss.append(before)
                    self.tower_after_loss.append(after)
                    self.tower_stop_token_loss.append(stop_token_loss)
                    self.tower_regularization_loss.append(regularization)
                    self.tower_linear_loss.append(linear_loss)

                    tower_loss = before + after + stop_token_loss + regularization + linear_loss
                    self.tower_loss.append(tower_loss)

            total_before_loss += before
            total_after_loss += after
            total_stop_token_loss += stop_token_loss
            total_regularization_loss += regularization
            total_linear_loss += linear_loss
            total_loss += tower_loss

        self.before_loss = total_before_loss / hp.tacotron_num_gpus
        self.after_loss = total_after_loss / hp.tacotron_num_gpus
        self.stop_token_loss = total_stop_token_loss / hp.tacotron_num_gpus
        self.regularization_loss = total_regularization_loss / hp.tacotron_num_gpus
        self.linear_loss = total_linear_loss / hp.tacotron_num_gpus
        self.loss = total_loss / hp.tacotron_num_gpus

    def add_optimizer(self, global_step):
        '''Adds optimizer. Sets "gradients" and "optimize" fields. add_loss must have been called.
        Args:
            global_step: int32 scalar Tensor representing current global step in training
        '''
        hp = self.hparams
        tower_gradients = []

        # 1. Declare GPU Devices
        gpus = ["/gpu:{}".format(i) for i in range(hp.tacotron_num_gpus)]

        grad_device = '/cpu:0' if hp.tacotron_num_gpus > 1 else gpus[0]

        with tf.device(grad_device):
            with tf.variable_scope('optimizer') as scope:
                if hp.tacotron_decay_learning_rate:
                    self.decay_steps = hp.tacotron_decay_steps
                    self.decay_rate = hp.tacotron_decay_rate
                    self.learning_rate = self._learning_rate_decay(hp.tacotron_initial_learning_rate, global_step)
                else:
                    self.learning_rate = tf.convert_to_tensor(hp.tacotron_initial_learning_rate)

                optimizer = tf.train.AdamOptimizer(self.learning_rate, hp.tacotron_adam_beta1,
                                                   hp.tacotron_adam_beta2, hp.tacotron_adam_epsilon)

        # 2. Compute Gradient
        for i in range(hp.tacotron_num_gpus):
            #  Device placement
            with tf.device(tf.train.replica_device_setter(ps_tasks=1, ps_device="/cpu:0", worker_device=gpus[i])):
                with tf.variable_scope('optimizer') as scope:
                    update_vars = [v for v in self.all_vars if not (
                            'inputs_embedding' in v.name or 'encoder_' in v.name)] if hp.tacotron_fine_tuning else None
                    gradients = optimizer.compute_gradients(self.tower_loss[i], var_list=update_vars)
                    tower_gradients.append(gradients)

        # 3. Average Gradient
        with tf.device(grad_device):
            avg_grads = []
            variables = []
            for grad_and_vars in zip(*tower_gradients):
                # each_grads_vars = ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
                grads = []
                for g, _ in grad_and_vars:
                    expanded_g = tf.expand_dims(g, 0)
                    # Append on a 'tower' dimension which we will average over below.
                    grads.append(expanded_g)
                # Average over the 'tower' dimension.

                grad = tf.concat(axis=0, values=grads)
                grad = tf.reduce_mean(grad, 0)

                v = grad_and_vars[0][1]
                avg_grads.append(grad)
                variables.append(v)

            self.gradients = avg_grads
            # Just for caution
            # https://github.com/Rayhane-mamah/Tacotron-2/issues/11
            if hp.tacotron_clip_gradients:
                clipped_gradients, _ = tf.clip_by_global_norm(avg_grads, 1.)  # __mark 0.5 refer
            else:
                clipped_gradients = avg_grads

            # Add dependency on UPDATE_OPS; otherwise batchnorm won't work correctly. See:
            # https://github.com/tensorflow/tensorflow/issues/1122
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                self.optimize = optimizer.apply_gradients(zip(clipped_gradients, variables),
                                                          global_step=global_step)

    # ####### declare variables
    # with tf.variable_scope('inference') as scope:
    #     batch_size = tf.shape(inputs)[0]
    #     hparams = self.hparams
    #     assert hparams.tacotron_teacher_forcing_mode in ('constant', 'scheduled')
    #     if hparams.tacotron_teacher_forcing_mode == 'scheduled' and is_training:
    #         assert global_step is not None
    #
    #
    #     ### get symbol to create embedding lookup table
    #     if self.hparams.hangul_type == 1:
    #         hangul_symbol = hangul_symbol_1
    #     elif self.hparams.hangul_type == 2:
    #         hangul_symbol = hangul_symbol_2
    #     elif self.hparams.hangul_type == 3:
    #         hangul_symbol = hangul_symbol_3
    #     elif self.hparams.hangul_type == 4:
    #         hangul_symbol = hangul_symbol_4
    #     else:
    #         hangul_symbol = hangul_symbol_5
    #
    #     # Embeddings ==> [batch_size, sequence_length, embedding_dim]
    #     # create embedding look up table with shape of [number of symbols, embedding dimension (declare in hparams]
    #     embedding_table = tf.get_variable(
    #         'inputs_embedding', [len(hangul_symbol), hparams.embedding_dim], dtype=tf.float32)
    #     ### inputs is a tensor of sequence of IDs  (created using text_to_sequence)
    #     # which is loaded through feeder class (_meta_data variable) from train.txt in training_data folder
    #     ## embedded_input is a Tensor with same type with embedding_table Tensor
    #     embedded_inputs = tf.nn.embedding_lookup(embedding_table, inputs)
    #     ##########################
    #     ## create Encoder object##
    #     ##########################
    #     encoder_cell = Encoder.EncoderCell(
    #         EncoderConvolution=Encoder.EncoderConvolution(
    #             is_training = is_training,
    #             hparams = hparams,
    #             scope='encoder_convolutions'
    #         ),
    #         EncoderLSTM = Encoder.EncoderLSTM(
    #             is_training= is_training,
    #             size=256,
    #             zoneout=0.1,
    #             scope='encoder_lstm'
    #         )
    #     )
    #     # extract Encoder output
    #     encoder_outputs = encoder_cell(embedded_inputs, input_lengths=input_lengths)
    #
    #     # store convolution output shape for visualization
    #     enc_conv_output_shape = encoder_cell.conv_output_shape
    #
    #     ##########################
    #     ## create Decoder object##
    #     ##########################
    #     decoder_cell = Decoder.TacotronDecoderCell(
    #         prenet = Decoder.Prenet(
    #             is_training=is_training,
    #             layers_sizes=[256, 256],
    #             drop_rate=hparams.dropout_rate,
    #             scope='decoder_prenet'
    #         ),
    #         attention_mechanism = Decoder.LocationSensitiveAttention(
    #             num_units=hparams.attention_dim,
    #             memory=encoder_outputs,
    #             hparams=hparams,
    #             mask_encoder=True,
    #             memory_sequence_length=input_lengths,
    #             cumulate_weights=True
    #         ),
    #         rnn_cell = Decoder.DecoderRNN(
    #             is_training=is_training,
    #             layers=2,
    #             zoneout=hparams.zoneout_rate,
    #             scope='decoder_lstm'
    #         ),
    #         frame_projection = Decoder.FrameProjection(
    #             shape=hparams.num_mels*2,
    #             scope='linear_transform'
    #         ),
    #         stop_projection = Decoder.StopProjection(
    #             is_training=is_training or is_evaluating,
    #             shape=2,
    #             scope='stop_token_projection'
    #         ),
    #     )
    #     ##initiate the first state of decoder
    #     decoder_init_state = decoder_cell.zero_state(batch_size=batch_size, dtype=tf.float32)
    #     # Define the helper for our decoder
    #     if is_training or is_evaluating or GTA:
    #         self.helper = TacoTrainingHelper(batch_size, mel_targets, stop_token_targets, hparams, GTA, is_evaluating,
    #                                          global_step)
    #     else:
    #         self.helper = TacoTestHelper(batch_size, hparams)
    #
    #     # Only use max iterations at synthesis time
    #     max_iters = hparams.max_iters if not (is_training or is_evaluating) else None
    #     # Decode
    #     (frames_prediction, stop_token_prediction, _), final_decoder_state, _ = dynamic_decode(
    #         Decoder.CustomDecoder(decoder_cell, self.helper, decoder_init_state),
    #         impute_finished=False,
    #         maximum_iterations=max_iters,
    #         swap_memory=hparams.tacotron_swap_with_cpu)
    #
    #     # Reshape outputs to be one output per entry
    #     # ==> [batch_size, non_reduced_decoder_steps (decoder_steps * r), num_mels]
    #     decoder_output = tf.reshape(frames_prediction, [batch_size, -1, hparams.num_mels])
    #     stop_token_prediction = tf.reshape(stop_token_prediction, [batch_size, -1])
    #
    #     # Postnet
    #     postnet = Postnet.Postnet(is_training, hparams=hparams, scope='postnet_convolutions')
    #
    #     # Compute residual using post-net ==> [batch_size, decoder_steps * r, postnet_channels]
    #     residual = postnet(decoder_output)
    #
    #     # Project residual to same dimension as mel spectrogram
    #     # ==> [batch_size, decoder_steps * r, num_mels]
    #     residual_projection = Decoder.FrameProjection(hparams.num_mels, scope='postnet_projection')
    #     projected_residual = residual_projection(residual)
    #
    #     # Compute the mel spectrogram
    #     mel_outputs = decoder_output + projected_residual
    #
    #     # time-domain waveforms is only used for predicting mels to train wavenet vocoder\
    #     # so we omit post processing when doing GTA synthesis
    #     post_condition = hparams.predict_linear and not GTA
    #     if post_condition: ### if use linear prediction
    #         # Based on https://github.com/keithito/tacotron/blob/tacotron2-work-in-progress/models/tacotron.py
    #         # Post-processing Network to map mels to linear spectrograms using same architecture as the encoder
    #         post_processing_cell = Encoder.EncoderCell(
    #             Encoder.EncoderConvolution(is_training, hparams=hparams, scope='post_processing_convolutions'),
    #             Encoder.EncoderLSTM(is_training, size=hparams.enc_lstm_hidden_size,
    #                                 zoneout=hparams.zoneout_rate, scope='post_processing_LSTM'))
    #
    #         expand_outputs = post_processing_cell(mel_outputs)
    #         linear_outputs = Decoder.FrameProjection(hparams.num_freq, scope='post_processing_projection')(expand_outputs)
    #         self.linear_outputs = linear_outputs
    #         self.linear_targets = linear_targets
    #     # Grab alignments from the final decoder state
    #     alignments = tf.transpose(final_decoder_state.alignment_history.stack(), [1, 2, 0])
    #
    #     if is_training:
    #         self.ratio = self.helper._ratio
    #     self.inputs = inputs
    #     self.input_lengths = input_lengths
    #     self.decoder_output = decoder_output
    #     self.alignments = alignments
    #     self.stop_token_prediction = stop_token_prediction
    #     self.stop_token_targets = stop_token_targets
    #     self.mel_outputs = mel_outputs
    #     self.mel_targets = mel_targets
    #     self.targets_lengths = targets_lengths
    #
    #     log('Initialized Tacotron model. Dimensions (? = dynamic shape): ')
    #     log('  Train mode:               {}'.format(is_training))
    #     log('  Eval mode:                {}'.format(is_evaluating))
    #     log('  GTA mode:                 {}'.format(GTA))
    #     log('  Synthesis mode:           {}'.format(not (is_training or is_evaluating)))
    #     log('  embedding:                {}'.format(embedded_inputs.shape))
    #     log('  enc conv out:             {}'.format(enc_conv_output_shape))
    #     log('  encoder out:              {}'.format(encoder_outputs.shape))
    #     log('  decoder out:              {}'.format(decoder_output.shape))
    #     log('  residual out:             {}'.format(residual.shape))
    #     log('  projected residual out:   {}'.format(projected_residual.shape))
    #     log('  mel out:                  {}'.format(mel_outputs.shape))
    #     if post_condition:
    #         log('  linear out:               {}'.format(linear_outputs.shape))
    #     log('  <stop_token> out:         {}'.format(stop_token_prediction.shape))

    # def add_loss(self):
    #     '''Adds loss to the model. Sets "loss" field. initialize must have been called.'''
    #     with tf.variable_scope('loss') as scope:
    #         hp = self.hparams
    #
    #         if hp.mask_decoder:
    #             # Compute loss of predictions before postnet
    #             before = MaskedMSE(self.mel_targets, self.decoder_output, self.targets_lengths,
    #                                hparams=self.hparams)
    #             # Compute loss after postnet
    #             after = MaskedMSE(self.mel_targets, self.mel_outputs, self.targets_lengths,
    #                               hparams=self.hparams)
    #             # Compute <stop_token> loss (for learning dynamic generation stop)
    #             stop_token_loss = MaskedSigmoidCrossEntropy(self.stop_token_targets,
    #                                                         self.stop_token_prediction, self.targets_lengths,
    #                                                         hparams=self.hparams)
    #         else:
    #             # Compute loss of predictions before postnet
    #             before = tf.losses.mean_squared_error(self.mel_targets, self.decoder_output)
    #             # Compute loss after postnet
    #             after = tf.losses.mean_squared_error(self.mel_targets, self.mel_outputs)
    #             # Compute <stop_token> loss (for learning dynamic generation stop)
    #             stop_token_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
    #                 labels=self.stop_token_targets,
    #                 logits=self.stop_token_prediction))
    #
    #         if hp.predict_linear:
    #             # Compute linear loss
    #             # From https://github.com/keithito/tacotron/blob/tacotron2-work-in-progress/models/tacotron.py
    #             # Prioritize loss for frequencies under 2000 Hz.
    #             l1 = tf.abs(self.linear_targets - self.linear_outputs)
    #             n_priority_freq = int(2000 / (hp.sample_rate * 0.5) * hp.num_mels)
    #             linear_loss = 0.5 * tf.reduce_mean(l1) + 0.5 * tf.reduce_mean(l1[:, :, 0:n_priority_freq])
    #         else:
    #             linear_loss = 0.
    #
    #         # Compute the regularization weight
    #         if hp.tacotron_scale_regularization:
    #             reg_weight_scaler = 1. / (2 * hp.max_abs_value) if hp.symmetric_mels else 1. / (hp.max_abs_value)
    #             reg_weight = hp.tacotron_reg_weight * reg_weight_scaler
    #         else:
    #             reg_weight = hp.tacotron_reg_weight
    #
    #         # Get all trainable variables
    #         all_vars = tf.trainable_variables()
    #         regularization = tf.add_n([tf.nn.l2_loss(v) for v in all_vars
    #                                    if not ('bias' in v.name or 'Bias' in v.name)]) * reg_weight
    #
    #         # Compute final loss term
    #         self.before_loss = before
    #         self.after_loss = after
    #         self.stop_token_loss = stop_token_loss
    #         self.regularization_loss = regularization
    #         self.linear_loss = linear_loss
    #         self.loss = self.before_loss + self.after_loss + self.stop_token_loss + self.regularization_loss + self.linear_loss
    #
    # def add_optimizer(self, global_step):
    #     '''Adds optimizer. Sets "gradients" and "optimize" fields. add_loss must have been called.
    #
    #     Args:
    #         global_step: int32 scalar Tensor representing current global step in training
    #     '''
    #     with tf.variable_scope('optimizer') as scope:
    #         hp = self.hparams
    #         if hp.tacotron_decay_learning_rate:
    #             self.decay_steps = hp.tacotron_decay_steps
    #             self.decay_rate = hp.tacotron_decay_rate
    #             self.learning_rate = self._learning_rate_decay(hp.tacotron_initial_learning_rate, global_step)
    #         else:
    #             self.learning_rate = tf.convert_to_tensor(hp.tacotron_initial_learning_rate)
    #
    #         optimizer = tf.train.AdamOptimizer(self.learning_rate, hp.tacotron_adam_beta1,
    #                                            hp.tacotron_adam_beta2, hp.tacotron_adam_epsilon)
    #         gradients, variables = zip(*optimizer.compute_gradients(self.loss))
    #         self.gradients = gradients
    #         # Just for causion
    #         # https://github.com/Rayhane-mamah/Tacotron-2/issues/11
    #         clipped_gradients, _ = tf.clip_by_global_norm(gradients, 1.)
    #
    #         # Add dependency on UPDATE_OPS; otherwise batchnorm won't work correctly. See:
    #         # https://github.com/tensorflow/tensorflow/issues/1122
    #         with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
    #             self.optimize = optimizer.apply_gradients(zip(clipped_gradients, variables),
    #                                                       global_step=global_step)

    def _learning_rate_decay(self, init_lr, global_step):
        #################################################################
        # Narrow Exponential Decay:

        # Phase 1: lr = 1e-3
        # We only start learning rate decay after 50k steps

        # Phase 2: lr in ]1e-5, 1e-3[
        # decay reach minimal value at step 310k

        # Phase 3: lr = 1e-5
        # clip by minimal learning rate value (step > 310k)
        #################################################################
        hp = self.hparams
        # Compute natural exponential decay
        lr = tf.train.exponential_decay(init_lr,
                                        global_step - hp.tacotron_start_decay,  # lr = 1e-3 at step 50k
                                        self.decay_steps,
                                        self.decay_rate,  # lr = 1e-5 around step 310k
                                        name='lr_exponential_decay')

        # clip learning rate by max and min values (initial and final values)
        return tf.minimum(tf.maximum(lr, hp.tacotron_final_learning_rate), init_lr)
