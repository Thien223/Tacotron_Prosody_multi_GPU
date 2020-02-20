from datetime import datetime
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2' ### disable warning messages
import time
import tensorflow as tf
import traceback
from Utils import Infolog
from Utils.TextProcessing.TextPreprocessing import sequence_to_text
from Utils.AudioProcessing.AudioPreprocess import *
from Utils.Tacotron_feeder import Feeder
from TacotronModel.modules.Tacotron import Tacotron
from Utils.Plot import plot_alignment, plot_spectrogram
from Utils.Utils import ValueWindow
log = Infolog.log

def train(log_dir, args, hparams):
	
	##prepare folder and pretrained model (if there is)
	save_dir = os.path.join(log_dir, 'taco_pretrained/')
	checkpoint_path = os.path.join(save_dir, 'tacotron_model.ckpt')
	input_path = os.path.join(args.base_dir, args.tacotron_input)
	plot_dir = os.path.join(log_dir, 'plots')
	wav_dir = os.path.join(log_dir, 'wavs')
	mel_dir = os.path.join(log_dir, 'mel-spectrograms')
	eval_dir = os.path.join(log_dir, 'eval-dir')
	eval_plot_dir = os.path.join(eval_dir, 'plots')
	eval_wav_dir = os.path.join(eval_dir, 'wavs')
	os.makedirs(eval_dir, exist_ok=True)
	os.makedirs(plot_dir, exist_ok=True)
	os.makedirs(wav_dir, exist_ok=True)
	os.makedirs(mel_dir, exist_ok=True)
	os.makedirs(eval_plot_dir, exist_ok=True)
	os.makedirs(eval_wav_dir, exist_ok=True)
	
	
	
	# save log info
	log('Checkpoint path: {}'.format(checkpoint_path))
	log('Loading training data from: {}'.format(input_path))
	# Start by setting a seed for repeatability
	tf.set_random_seed(hparams.tacotron_random_seed)
	# Set up data feeder
	## create an object of Feeder class to feed preprocessed data:
	#  (audio time series, mel spectrogram matrix, text sequences) to training model
	coord = tf.train.Coordinator()
	with tf.variable_scope('datafeeder'):
		feeder = Feeder(coord, input_path, hparams)
	
	########################################
	# Set up model:
	# create Tacotron model
	global_step = tf.Variable(0, name='global_step', trainable=False) ## define global step to use in tf.train.cosine_decay() when using teacher forcing
	model = initiallize_model_variables(feeder, hparams, global_step)
	
	### context variables, hold current information
	step = 0
	time_window = ValueWindow(100)
	loss_window = ValueWindow(100)
	## saver to save model checkpoint.
	saver = tf.train.Saver(max_to_keep=5)
	
	log('Tacotron training set to a maximum of {} steps'.format(args.tacotron_train_steps))
	
	# Memory allocation on the GPU as needed
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	config.gpu_options.per_process_gpu_memory_fraction=0.7
	config.allow_soft_placement=True
	# Train
	with tf.Session(config=config) as sess:
		try:
			sess.run(tf.global_variables_initializer())
			# restore saved model
			if args.restore:
				# Restore saved model if the user requested it, Default = True.
				try:
					checkpoint_state = tf.train.get_checkpoint_state(save_dir)
				except tf.errors.OutOfRangeError as e:
					log('Cannot restore checkpoint: {}'.format(e))
			## if restoring is success
			if (checkpoint_state and checkpoint_state.model_checkpoint_path):
				log('Loading checkpoint {}'.format(checkpoint_state.model_checkpoint_path))
				saver.restore(sess, checkpoint_state.model_checkpoint_path)
			### if restoring is failed
			else:
				if not args.restore:
					log('Starting new training!')
				else:
					log('No model to load at {}'.format(save_dir))
			## feed preprocessed data to threads
			feeder.start_threads(sess)
			# Training loop
			while not coord.should_stop() and step < args.tacotron_train_steps:
				start_time = time.time()
				step, loss, opt = sess.run([global_step, model.loss, model.optimize])
				
				### save current infor and print to console
				time_window.append(time.time() - start_time)
				loss_window.append(loss)
				message = 'Step {:7d} [{:.3f} sec/step, loss={:.5f}, avg_loss={:.5f}]'.format(
					step, time_window.average, loss, loss_window.average)
				log(message, end='\r')
				if np.isnan(loss):
					log('Loss exploded to {:.5f} at step {}'.format(loss, step))
					raise Exception('Loss exploded')
				
				
				##### save check point when meeting checkpoint interval
				if step % args.checkpoint_interval == 0 or step == args.tacotron_train_steps:
					# Save model and current global step
					saver.save(sess, checkpoint_path, global_step=global_step)
					log('\nSaving Mel-Spectrograms..')
					input_seq, mel_prediction, alignment, target, target_length = sess.run([model.tower_inputs[0][0],
					                                                                        model.tower_mel_outputs[0][0],
					                                                                        model.tower_alignments[0][0],
					                                                                        model.tower_mel_targets[0][0],
					                                                                        model.tower_targets_lengths[0][0]
					                                                                        ])
					# save predicted mel spectrogram to disk (debug)
					mel_filename = 'mel-prediction-step-{}.npy'.format(step)
					np.save(os.path.join(mel_dir, mel_filename), mel_prediction.T, allow_pickle=False)
					
					# save griffin lim inverted wav for debug (mel -> wav)
					wav = mel_to_audio_serie(mel_prediction.T, hparams)
					if hparams.preemphasize:
						wav = inv_preemphasis(wav, hparams.preemphasis)
					save_wav(wav, os.path.join(wav_dir, 'step-{}-wave-from-mel.wav'.format(step)),
					         sr=hparams.sample_rate)
					
					# save alignment plot to disk (control purposes)
					plot_alignment(alignment, os.path.join(plot_dir, 'step-{}-align.png'.format(step)),
					               info='{}, {}, step={}, loss={:.5f}'.format(args.model, datetime.now().strftime('%Y-%m-%d %H:%M'), step,
					                                                          loss),
					               max_len=target_length // hparams.outputs_per_step)
					# save real and predicted mel-spectrogram plot to disk (control purposes)
					plot_spectrogram(mel_prediction,
					                 os.path.join(plot_dir, 'step-{}-mel-spectrogram.png'.format(step)),
					                 info='{}, {}, step={}, loss={:.5}'.format(args.model, datetime.now().strftime('%Y-%m-%d %H:%M'), step,
					                                                           loss), target_spectrogram=target,
					                 max_len=target_length)
					log('Input at step {}: {}'.format(step, sequence_to_text(input_seq)))
					##### FINISH training
					##### Testing....
					## do the test when step is the maximum of tacotron training step
			return save_dir
		except Exception as e:
			log('Exiting due to exception: {}'.format(e))
			traceback.print_exc()
			coord.request_stop(e)


def initiallize_model_variables(feeder, hparams, global_step):
	'''
	Initiallize values of model in training mode
	:param args: run time params
	:param feeder: data feeder object
	:param hparams: hyper params
	:param global_step: training step
	:return: initiallized model
	'''
	with tf.variable_scope('model', reuse=tf.AUTO_REUSE) as scope:
		#create Tacotron model object
		model = Tacotron(hparams)
		#initialize Tacotron model parameters based on type of predict (linear or not)
		model.initialize(inputs=feeder.inputs,
		                 input_lengths=feeder.input_lengths,
		                 mel_targets=feeder.mel_targets,
		                 stop_token_targets=feeder.token_targets,
		                 targets_lengths=feeder.targets_lengths,
		                 global_step=global_step,
		                 is_training=True,
		                 split_infos=feeder.split_infos)
		## add loss function to model
		model.add_loss()
		## optimizer
		model.add_optimizer(global_step)
		return model


def get_arguments():
	parser = argparse.ArgumentParser()
	parser.add_argument('--base_dir', default='')
	parser.add_argument('--hparams', default='',
	                    help='Hyperparameter overrides as a comma-separated list of name=value pairs')
	parser.add_argument('--tacotron_input', default='Tacotron_input/train.txt')
	parser.add_argument('--model', default='Tacotron')
	parser.add_argument('--output_dir', default='tacotron_output/', help='folder to contain synthesized mel spectrograms')
	parser.add_argument('--mode', default='synthesis', help='mode for synthesis of tacotron after training') ### synthesis, tune, inference
	parser.add_argument('--GTA', default='True',
	                    help='Ground truth aligned synthesis, defaults to True, only considered in Tacotron synthesis mode')
	parser.add_argument('--restore', type=bool, default=True, help='Set this to False to do a fresh training')
	parser.add_argument('--checkpoint_interval', type=int, default=1000,
	                    help='Steps between writing checkpoints')
	parser.add_argument('--tacotron_train_steps', type=int, default=2000000,
	                    help='total number of tacotron training steps')
	args = parser.parse_args()
	return args


if __name__ == '__main__':
	args = get_arguments()
	modified_hparams = hparams.parse(args.hparams) ### update hyperparam using args.hparams variables
	train('Tacotron_trained_logs',args, hparams=modified_hparams)
	tf.reset_default_graph()