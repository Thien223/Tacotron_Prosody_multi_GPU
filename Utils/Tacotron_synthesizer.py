import os
from platform import system
import numpy as np
import tensorflow as tf
from TacotronModel.modules.Tacotron import Tacotron
from Utils.AudioProcessing.AudioPreprocess import mel_to_audio_serie, save_wav, inv_preemphasis
from Utils.Infolog import log
from Utils.Plot import plot_spectrogram, plot_alignment
from Utils.Tacotron_feeder import _prepare_inputs, _prepare_targets, _get_output_lengths
from Utils.TextProcessing.HangulUtils import hangul_to_sequence


class Synthesizer:
    def load(self, checkpoint_path, hparams, GTA=False, reference_mel=None, model_name='Tacotron'):
        log('Constructing model: %s' % model_name)
        inputs = tf.placeholder(tf.int32, (None, None), name='inputs')
        input_lengths = tf.placeholder(tf.int32, (None), name='input_lengths')
        targets = tf.placeholder(tf.float32, [None, None, hparams.num_mels], 'mel_targets')

        targets_lengths = tf.placeholder(tf.int32, (None, hparams.num_mels), name='targets_lengths')
        if reference_mel is not None:
            reference_mel = tf.placeholder(tf.float32, [1, None, hparams.num_mels], 'reference_mel')

        split_infos = tf.placeholder(tf.int32, shape=(hparams.tacotron_num_gpus, None), name='split_infos')
        with tf.variable_scope('model'):
            self.model = Tacotron(hparams)
            if GTA:
                self.model.initialize(inputs=inputs, input_lengths=input_lengths, mel_targets=targets, GTA=GTA, split_infos=split_infos,reference_mel=reference_mel)
            else:
                self.model.initialize(inputs=inputs, input_lengths=input_lengths, split_infos=split_infos,reference_mel=reference_mel)
            self.mel_outputs = self.model.tower_mel_outputs
            self.alignment = self.model.tower_alignments
            self.stop_token = self.model.tower_stop_token_prediction
            self.targets = targets
            self.encoder_outputs = self.model.encoder_outputs

        self.GTA = GTA
        self.hparams = hparams

        log('Loading checkpoint: %s' % checkpoint_path)
        # Memory allocation on the GPUs as needed
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True

        self.session = tf.Session(config=config)

        self.session.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(self.session, checkpoint_path)


        self.inputs = inputs
        self.input_lengths = input_lengths
        self.mel_targets = targets
        self.split_infos = split_infos

    ### synthesize : synth.synthesize(texts, i+1, synth_dir, None, mel_filenames)
    ### inference: synth.synthesize(text, i + 1, inference_dir, log_dir, None)
    def synthesize(self, texts, basenames, out_dir, log_dir, mel_filenames):
        
        hparams = self.hparams

        # Repeat last sample until number of samples is dividable by the number of GPUs (last run scenario)
        while len(texts) % hparams.tacotron_num_gpus != 0:
            texts.append(texts[-1])
            basenames.append(basenames[-1])
            if mel_filenames is not None:
                mel_filenames.append(mel_filenames[-1])
        assert 0 == len(texts) % self.hparams.tacotron_num_gpus
        seqs = [np.asarray(hangul_to_sequence(dir=hparams.base_dir, hangul_text=text, hangul_type=hparams.hangul_type)) for text in texts]
        input_lengths = [len(seq) for seq in seqs]
        ## calculate sequence length on each GPU device
        sequence_size_per_device = len(seqs)// hparams.tacotron_num_gpus

        ##### create input sequence on each GPU then concat
        split_infos = []

        input_sequence = None ### synthesize input sequence
        for i in range(hparams.tacotron_num_gpus):
            on_device_input_sequence = seqs[sequence_size_per_device*i : sequence_size_per_device*(i+1)]
            on_device_input_sequence_padded, input_length = _prepare_inputs(inputs = on_device_input_sequence)
            input_sequence = np.concatenate((input_sequence, on_device_input_sequence_padded), axis=1) if input_sequence is not None else on_device_input_sequence_padded
            split_infos.append([input_length, 0,0,0])

        ### add input info to feed dict
        feed_dict = {
            self.inputs: input_sequence,
            self.input_lengths: np.asarray(input_lengths, dtype=np.int32)
        }

        if self.GTA:
            np_targets = [np.load(mel_filename) for mel_filename in mel_filenames]
            assert len(np_targets) == len(texts)

            #### get target sequence from mel_targets on each GPU
            target_sequence = None  ### synthesize input sequence
            for i in range(hparams.tacotron_num_gpus):
                on_device_target_sequence = np_targets[sequence_size_per_device * i: sequence_size_per_device * (i + 1)]
                on_device_target_sequence_padded, target_length = _prepare_targets(targets=on_device_target_sequence, alignment=hparams.outputs_per_step)
                target_sequence = np.concatenate((target_sequence, on_device_target_sequence_padded), axis=1) if target_sequence is not None else on_device_target_sequence_padded
                split_infos[i][1] = target_length


            ## add target  mel sequence to feed dict
            feed_dict[self.targets] = target_sequence
        feed_dict[self.split_infos]= np.asarray(split_infos, dtype=np.int32)

        ####### synthesize #######
        mels, alignments, stop_tokens,encoder_outputs = self.session.run([self.mel_outputs, self.alignment, self.stop_token,self.encoder_outputs], feed_dict=feed_dict)
        # Linearised outputs (n_gpus -> 1D)
        mels = [mel for gpu_mels in mels for mel in gpu_mels]
        
        alignments = [align for gpu_aligns in alignments for align in gpu_aligns]

        stop_tokens = [token for gpu_token in stop_tokens for token in gpu_token]
        # for i,seq in enumerate(seqs):
        #     print(feed_dict[self.inputs][i])
        #     print(len(seq))
        #     print(texts[i])
        #     print('=============================================')
        #
        # print([max(stop_token) for stop_token in stop_tokens])
        # print([len(stop_token) for stop_token in stop_tokens])
        # print(feed_dict[self.input_lengths])
        target_lengths = _get_output_lengths(stop_tokens)
        ##todo: need more effort this code part

        # cut off the silence part (the part behind stop_token)
        mels = [mel[:target_length, :] for mel, target_length in zip(mels, target_lengths)]

        # [-max, max] or [0,max]
        T2_output_range = (-hparams.max_abs_value, hparams.max_abs_value) if hparams.symmetric_mels else (0, hparams.max_abs_value)
        
        mels = [np.clip(mel, T2_output_range[0], T2_output_range[1]) for mel in mels]
        
        if basenames is None:
            # Generate wav and read it
            wav = mel_to_audio_serie(mels[0].T, hparams)
            save_wav(wav, 'temp.wav', sr=hparams.sample_rate)  # Find a better way
            if system() == 'Linux':
                # Linux wav reader
                os.system('aplay temp.wav')
            elif system() == 'Windows':
                # windows wav reader
                os.system('start /min mplay32 /play /close temp.wav')
            else:
                raise RuntimeError('Your OS type is not supported yet, please add it to "tacotron/synthesizer.py, line-150" and feel free to make a Pull Request ;) Thanks!')
            return
        
        saved_mels_paths = []
        speaker_ids=[]
        for (i,mel), text in zip(enumerate(mels), texts):
            # Get speaker id for global conditioning (only used with GTA generally)
            if hparams.gin_channels > 0:
                ### when there are many speakers (need edit this code part for multiple speakers model
                speaker_id = '<no_g>'  # set the rule to determine speaker id. By using the file basename maybe? (basenames are inside "basenames" variable)
                speaker_ids.append(speaker_id)  # finish by appending the speaker id. (allows for different speakers per batch if your model is multispeaker)
            else:
                ### when only 1 speaker
                speaker_id = '<no_g>'
                speaker_ids.append(speaker_id)
                
            # Write the spectrogram to disk
            # Note: outputs mel-spectrogram files and target ones have same names, just different folders
            mel_filename = os.path.join(out_dir, 'speech-mel-{}.npy'.format(basenames[i]))
            np.save(mel_filename, mel, allow_pickle=False)
            saved_mels_paths.append(mel_filename)

            if log_dir is not None:
                # save wav (mel -> wav)
                wav = mel_to_audio_serie(mel.T, hparams)
                wav = inv_preemphasis(wav, hparams.preemphasis, hparams.preemphasize)
                save_wav(wav, os.path.join(log_dir, 'wavs/speech-wav-{}-mel.wav'.format(basenames[i])),
                         sr=hparams.sample_rate)
                # save alignments
                plot_alignment(alignments[i], os.path.join(log_dir, 'plots/speech-alignment-{}.png'.format(basenames[i])),
                               info='{}'.format(texts[i]), split_title=True)
                # save mel spectrogram plot
                plot_spectrogram(mel, os.path.join(log_dir, 'plots/speech-mel-{}.png'.format(basenames[i])),
                                 info='{}'.format(texts[i]), split_title=True)
        return saved_mels_paths, speaker_ids

