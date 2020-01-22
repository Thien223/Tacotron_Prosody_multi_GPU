import os
import argparse
import tensorflow as tf
from tqdm import tqdm
from Utils.Infolog import log
from Utils.Tacotron_synthesizer import Synthesizer
from Utils.Hyperparams import hparams

def run_synthesis(args, checkpoint_path, output_dir, hparams):
    '''
    generate mel spectrograms from text using trained model
    :param args: run time params
    :param checkpoint_path: path to checkpoint of pretrained model
    :param output_dir: output dir to save spectrograms (can be got from args)
    :param hparams: Hyper params
    :return:
    '''

    GTA = (args.GTA == 'True')
    if GTA:
        synth_dir = os.path.join(output_dir, 'gta')
        # Create output path if it doesn't exist
        os.makedirs(synth_dir, exist_ok=True)
    else:
        synth_dir = os.path.join(output_dir, 'natural')
        # Create output path if it doesn't exist
        os.makedirs(synth_dir, exist_ok=True)

    metadata_filename = os.path.join(args.input_dir, 'train.txt')
    synth = Synthesizer()
    synth.load(checkpoint_path, hparams, GTA=GTA)


    ### read data from train.txt file <-- this file is generated after preprocessing
    with open(metadata_filename, encoding='utf-8') as f:
        metadata = [line.strip().split('|') for line in f]
        frame_shift_ms = hparams.hop_size / hparams.sample_rate
        hours = sum([int(x[4]) for x in metadata]) * frame_shift_ms / (3600)
        log('Loaded metadata for {} examples ({:.2f} hours)'.format(len(metadata), hours))

    ## generate batches from metadata
    metadata = [metadata[i: i + 512] for i in range(0, len(metadata), 512)] ### fix hparams.tacotron_synthesis_batch_size = 512
    log('starting synthesis..')
    mel_dir = os.path.join(args.input_dir, 'mels')
    wav_dir = os.path.join(args.input_dir, 'audio')

    #### batch synthesizing. Need  more effort
    with open(os.path.join(synth_dir, 'map.txt'), 'w') as file:
        for i, meta in enumerate(tqdm(metadata)):
            texts = [m[5] for m in meta]
            mel_filenames = [os.path.join(mel_dir, m[1]) for m in meta]
            wav_filenames = [os.path.join(wav_dir, m[0]) for m in meta]
            basenames = [os.path.basename(m).replace('.npy', '').replace('mel-', '') for m in mel_filenames]
            mel_output_filenames, speaker_ids = synth.synthesize(texts, basenames, synth_dir, None, mel_filenames)
            for elems in zip(wav_filenames, mel_filenames, mel_output_filenames, speaker_ids, texts):
                file.write('|'.join([str(x) for x in elems]) + '\n')

    # with open(os.path.join(synth_dir, 'map.txt'), 'w') as file:
    #     for i, meta in enumerate(tqdm(metadata)):
    #         text = meta[5]
    #         mel_filename = os.path.join(mel_dir, meta[1])
    #         wav_filename = os.path.join(wav_dir, meta[0])
    #         mel_output_filename, speaker_id = synth.synthesize(text, i + 1, synth_dir, None, mel_filename)
    #         file.write('{}|{}|{}|{}\n'.format(wav_filename, mel_filename, mel_output_filename, text))
    log('Predicted mel spectrograms are saved in {}'.format(synth_dir))
    return os.path.join(synth_dir, 'map.txt')

def run_inference(checkpoint_path, output_dir, hparams, sentences):
    inference_dir = os.path.join(output_dir, 'inference')
    log_dir = os.path.join(output_dir, 'logs-inference')
    os.makedirs(inference_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(os.path.join(log_dir, 'wavs'), exist_ok=True)
    os.makedirs(os.path.join(log_dir, 'plots'), exist_ok=True)
    log('running inference..')
    synth = Synthesizer()
    synth.load(checkpoint_path, hparams, GTA=False)
    sentences = [sentences[i: i + hparams.tacotron_synthesis_batch_size] for i in
                 range(0, len(sentences), hparams.tacotron_synthesis_batch_size)]
    ### save synthesized info to map.txt and folder
    with open(os.path.join(inference_dir, 'map.txt'), 'w') as file:
        for i, text in enumerate(tqdm(sentences)):
            basenames = ['batch_{}_sentence_{}'.format(i, j) for j in range(len(text))]
            mel_filename = synth.synthesize(text, basenames, inference_dir, log_dir, None)
            file.write('{}|{}\n'.format(text, mel_filename))
    log('synthesized mel spectrograms of \"{}\" at {}'.format(sentences,inference_dir))

    return inference_dir

def tacotron_synthesize(args, hparams, checkpoint):
    output_dir = args.output_dir
    try:
        checkpoint_path = tf.train.get_checkpoint_state(checkpoint).model_checkpoint_path
        log('loaded model at {}'.format(checkpoint_path))
    except AttributeError:
            raise RuntimeError('Failed to load checkpoint at {}'.format(checkpoint))
    return run_synthesis(args, checkpoint_path, output_dir, hparams)


def tacotron_inference(args, hparams, checkpoint, sentences):
    output_dir = args.output_dir
    if sentences is None:
        raise RuntimeError('Inference mode requires input sentence(s), make sure you put sentences in sentences.txt!')

    try:
        checkpoint_path = tf.train.get_checkpoint_state(checkpoint).model_checkpoint_path
        log('loaded model at {}'.format(checkpoint_path))
    except AttributeError:
        raise RuntimeError('Failed to load checkpoint at {}'.format(checkpoint))
    return run_inference(checkpoint_path, output_dir, hparams, sentences)

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='Tacotron_trained_logs/taco_pretrained/', help='Path to model checkpoint')
    parser.add_argument('--hparams', default='',
                        help='Hyperparameter overrides as a comma-separated list of name=value pairs')
    parser.add_argument('--name', help='Name of logging directory if the two modules were trained together.')
    parser.add_argument('--tacotron_name', help='Name of logging directory of Tacotron. If trained separately')
    parser.add_argument('--model', default='Tacotron-2')
    parser.add_argument('--mode', default='synthesize', help='runing mode, could be synthesize or inference')
    parser.add_argument('--input_dir', default='Tacotron_input/', help='folder to contain inputs sentences/targets')
    parser.add_argument('--output_dir', default='tacotron_output/', help='folder to contain synthesized mel spectrograms')
    parser.add_argument('--GTA', default='True',
                        help='Ground truth aligned synthesis, defaults to True, only considered in synthesis mode')
    parser.add_argument('--speaker_id', default=None, help='speaker ids list, comma separated')
    args = parser.parse_args()
    return args

def get_sentences(hparams):
    '''
    Get desire sentences from sentences.txt. If the file is empty, then load sentences from hparams.
    :param hparams: holds alternative sentences.
    :return: sentences list
    '''
    with open('sentences.txt', 'rb') as f:
        sentences = list(map(lambda l: l.decode("utf-8")[:-1], f.readlines()))
    if len(sentences) != 0:
        return sentences
    else:
        sentences = hparams.sentences
        return sentences


if __name__ == '__main__':
    args=get_arguments()
    modified_hparams = hparams.parse(args.hparams)  ### update hyperparam using args.hparams variables
    if args.mode =='synthesize':
        out_dir = tacotron_synthesize(args, modified_hparams, args.checkpoint)

    elif args.mode =='inference':
        sentences=get_sentences(modified_hparams) ### get sentences from sentences.txt file (or in hparams)
        inference_dir = tacotron_inference(args,modified_hparams,args.checkpoint,sentences)
        ### join audio files into one file
        if hparams.join_output_audio:
            inference_dir=inference_dir.replace('map.txt','').strip()
            from natsort import natsorted
            import glob
            os.makedirs(inference_dir+'/joined/',exist_ok=True)
            audio_segment_list = []
            files = glob.glob(inference_dir+'/wavs/*.wav')
            ## sort files to make sure they are in right order
            files = natsorted(files)
            from pydub import AudioSegment
            combined=AudioSegment.empty()
            for file in files:
                temp_wav = AudioSegment.from_wav(file)
                audio_segment_list.extend(temp_wav)
            for audio in audio_segment_list:
                combined = combined+audio
            combined.export(inference_dir+'/joined/joined_output.wav',format='wav')

