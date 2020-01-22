import tensorflow as tf
import numpy as np
# Default hyperparameters
hparams = tf.contrib.training.HParams(

    #Global style token
    use_gst=True,     # When false, the scripit will do as the paper  "Towards End-to-End Prosody Transfer for Expressive Speech Synthesis with Tacotron"
    num_gst=10,
    num_heads=4,       # Head number for multi-head attention
    style_embed_depth=256,
    reference_filters=[32, 32, 64, 64, 128, 128],
    reference_depth=128,
    style_att_type="mlp_attention", # Attention type for style attention module (dot_attention, mlp_attention)
    style_att_dim=128,


    #Loss and optimizing params
    tacotron_adam_beta1=0.9,  # AdamOptimizer beta1 parameter
    tacotron_adam_beta2=0.999,  # AdamOptimizer beta2 parameter
    tacotron_adam_epsilon=1e-6,  # AdamOptimizer beta3 parameter
    mask_decoder = False,
    leaky_alpha=0.4,
    mask_encoder = True,

    #### teacher forcing parameters
    tacotron_decay_learning_rate=True,
    tacotron_start_decay=40000,  # Step at which learning decay starts
    tacotron_decay_steps=18000,  # Determines the learning rate decay slope (UNDER TEST)
    tacotron_decay_rate=0.5,  # learning rate decay rate (UNDER TEST)
    tacotron_initial_learning_rate=1e-3,  # starting learning rate
    tacotron_final_learning_rate=1e-4,  # minimal learning rate
    tacotron_teacher_forcing_mode = 'constant',    # mode of teacher forcing Can be ('constant' or 'scheduled'). 'scheduled' mode applies a cosine teacher forcing ratio decay. (Preference: scheduled)
    tacotron_teacher_forcing_ratio=1., # Value from [0., 1.], 0.=0%, 1.=100%, determines the % of times we force next decoder inputs, Only relevant if mode='constant'
    tacotron_teacher_forcing_init_ratio=1.,  # initial teacher forcing ratio. Relevant if mode='scheduled'
    tacotron_teacher_forcing_final_ratio=0.,  # final teacher forcing ratio. Relevant if mode='scheduled'
    tacotron_teacher_forcing_start_decay=10000,# starting point of teacher forcing ratio decay. Relevant if mode='scheduled'
    tacotron_teacher_forcing_decay_steps=40000,# Determines the teacher forcing ratio decay slope. Relevant if mode='scheduled'
    tacotron_teacher_forcing_decay_alpha=None,  # teacher forcing ratio decay rate. Relevant if mode='scheduled'
    tacotron_reg_weight=1e-6,  # regularization weight (for L2 regularization)
    tacotron_scale_regularization=False,

    ##### network params
    conv_num_layers = 3,
    conv_kernel_size = (5,), # convolution filter size
    conv_channels = 512,        #convolution channels (encoder output dims)
    enc_lstm_hidden_size = 256, #lstm hiddent units size
    enc_conv_layers = 5,         #number of layers of convolutional network
    enc_conv_droprate=0.5,      #convolutional network droprate
    zoneout_rate = 0.1,         #zoneout rate for zoneoutLSTM network
    dropout_rate = 0.5,

    smoothing=False,
    attention_dim =128,
    attention_filters = 32,
    attention_kernel=(31,),
    cumulative_weights = True, #Whether to cumulate (sum) all previous attention weights or simply feed previous weights (Recommended: True)

    embedding_dim=512,          # dimensions of embedded vector
    postnet_kernel_size=(5,),   #postnet convolutional layers filter size
    postnet_channels = 512,     #postnet convolution channels of each layer
    postnet_num_layers = 5,     #number of postnet convolutional layers
	prenet_layers = [256, 256], #number of layers and number of units of prenet
    decoder_lstm_units = 1024,
    #text preprocessing params
    hangul_type = 1,        #type of hangul conversion
    ##audio preprocessing params
    trim_fft_size = 512,    ## use to trim silent part of M-AILABS dataset
    trim_hop_size = 128,    # use to trim silent part of M-AILABS dataset
    trim_top_db = 60,   # use to trim silent part of M-AILABS dataset
    rescale=True,  # whether rescale audio data before processing
    rescaling_max = 0.999,    #max scaling (if rescale is true, this parameter will be used)
    trim_silence = True,        #whether trim out silent parts
    sample_rate=22050, #22050, 44100
    fmin=25,  # min of voice frequency
    fmax=7600,  # max of voice frequency
    min_level_db=-100,
    ref_level_db=20,
    silence_threshold = 2,  #threshold of tobe trimmed silence audio
    wavenet_weight_normalization = False,
    #linear spectrogram params
    power=1.2,
    griffin_lim_iters=60,
    ### mel spectrogram params
    predict_linear = False,         #whether model predicts linear spectrogram
    signal_normalization=True,
    allow_clipping_in_normalization=True,  # Only relevant if mel_normalization = True
    symmetric_mels=True,  # Whether to scale the data to be symmetric around 0
    max_abs_value=4.,  # max absolute value of data. If symmetric, data will be [-max, max] else [0, max]
    num_mels=80,  # Number of mel-spectrogram channels and local conditioning dimensionality
    num_freq=1025,  # (= n_fft / 2 + 1) only used when adding linear spectrograms post processing network
    clip_mels_length=True,  # For cases of OOM (Not really recommended, working on a workaround)
    max_mel_frames=1000,  # Only relevant when clip_mels_length = True
    n_fft=2048,  # Extra window size is filled with 0 padding to match this parameter
    hop_size=275,  # For 22050Hz, 275 ~= 12.5 ms
    win_size=1024,  # For 22050Hz, 1100 ~= 50 ms (If None, win_size = n_fft) keep win_size/sample_rate ratio at 20~40 ms
    frame_shift_ms=None,
    ##audio preprocessing params
    preemphasize=False,  # whether to apply filter
    preemphasis=0.97,  # filter coefficient.
    ### Global Style Tokens params
    attention_heads=4,
    token_embedding_size=256/4, ### = 256/attention_heads
    num_tokens=10,
    speakers_path = None,
    speakers = ['speaker0', 'speaker1', #List of speakers used for embeddings visualization. (Consult "Wavenet_vocoder/train.py" if you want to modify the speaker names source).
				'speaker2', 'speaker3', 'speaker4'],
    # wavenet paramters0
    # Input type:
    # 1. raw [-1, 1]
    # 2. mulaw [-1, 1]
    # 3. mulaw-quantize [0, mu]
    # If input_type is raw or mulaw, network assumes scalar input and
    # discretized mixture of logistic distributions output, otherwise one-hot

    # Model Losses parmeters
    # Minimal scales ranges for MoL and Gaussian modeling
    cdf_loss = False, #Whether to use CDF loss in Gaussian modeling. Advantages: non-negative loss term and more training stability. (Automatically True for MoL)

    # input and softmax output are assumed.
    input_type="raw",
    #if input_type = mulaw_quantize then:
    quantize_channels=2**16,
    legacy = True, #Whether to use legacy mode: Multiply all skip outputs but the first one with sqrt(0.5) (True for more early training stability, especially for large modules)
	residual_legacy = True, #Whether to scale residual blocks outputs by a factor of sqrt(0.5) (True for input variance preservation early in training and better overall stability)

    #lws: local weighted sum
    use_lws=False,
    cin_channels= 80, ### requires cin_channel = num_mels
    #number of dilated convolutional network layers
    layers = 20, #######################################################################################################################
    # group convolutional layers to [number] of stack
    stacks = 2,   #######################################################################################################################
    out_channels= 2, ## for raw
    # out_channels= 10 * 3, ## for mulaw
    # out_channels= 255, ## for mulaw-quantized
    residual_channels=128,
    gate_channels = 256,
    kernel_size = 3,
    skip_out_channels = 128,
    wavenet_dropout = 0.05,
    use_bias = True,
    gin_channels = -1,
    max_time_sec=None,
    max_time_steps=10000,  # Max time steps in audio used to train wavenet (decrease to save memory)
    upsample_conditional_features=True,# Whether to repeat conditional features or upsample them (The latter is recommended)
    upsample_scales=[11,25],  # prod(scales) should be equal to hop size

    # Type of the upsampling deconvolution. Can be ('1D' or '2D', 'Resize', 'SubPixel' or simple 'NearestNeighbor').
    freq_axis_kernel_size=3,
    log_scale_min=float(np.log(1e-14)),

    log_scale_min_gauss=float(np.log(1e-7)),  # Gaussian distribution minimal allowed log scale

    normalize_for_wavenet=True,
#training, evaluating, synthesizing params
    #Tacotron training params
    tacotron_num_gpus = 1,
    split_on_cpu = True,
    tacotron_synthesis_batch_size = 1,
    tacotron_data_random_state = 1324,
    outputs_per_step = 2,
    tacotron_random_seed=45454,
    tacotron_swap_with_cpu = False,
    tacotron_test_size=0.1,
    tacotron_test_batches=None,
    tacotron_batch_size=48,
    max_iters = 3000,
    stop_at_any=True,
    clip_outputs=True,
    lower_bound_decay = 0.1,
    tacotron_fine_tuning=False, ## change to True when tuning other voice
    tacotron_clip_gradients = True, #whether to clip gradients
    # wavenet Training params
    wavenet_synthesis_batch_size = 3 * 2,
    wavenet_random_seed=5339,  # S=5, E=3, D=9 :)
    wavenet_swap_with_cpu=False, # Whether to use cpu as support to gpu for decoder computation
    wavenet_batch_size=3,  # batch size used to train wavenet.
    wavenet_test_size=0.0441,  # % of data to keep as test data, if None, wavenet_test_batches must be not None
    wavenet_test_batches=None,  # number of test batches.
    wavenet_data_random_state=1234,  # random state for train test split repeatability
    wavenet_warmup = float(4000), #Only used with 'noam' scheme. Defines the number of ascending learning rate steps.
    wavenet_learning_rate=1e-3,
    wavenet_adam_beta1=0.9,
    wavenet_adam_beta2=0.999,
    wavenet_adam_epsilon=1e-6,
    wavenet_ema_decay=0.9999,  # decay rate of exponential moving average
    train_with_GTA=False,  # Whether to use GTA mels to train wavenet instead of ground truth mels.

    # Learning rate schedule
    wavenet_lr_schedule='exponential',  # learning rate schedule. Can be ('exponential', 'noam')
    wavenet_decay_rate=0.5,  # Only used with 'exponential' scheme. Defines the decay rate.
    wavenet_decay_steps=200000,  # Only used with 'exponential' scheme. Defines the decay steps.

    # Optimization parameters
    #Evaluation parameters
	wavenet_natural_eval = False, #Whether to use 100% natural eval (to evaluate autoregressivity performance) or with teacher forcing to evaluate overfit and model consistency.

    # Regularization parameters
    wavenet_clip_gradients=True,  # Whether the clip the gradients during wavenet training.
    wavenet_init_scale=1.,
    wavenet_gradient_max_norm=100.0,  # Norm used to clip wavenet gradients
    wavenet_gradient_max_value=5.0,  # Value used to clip wavenet gradients

    ## environment params
    base_dir='',
    join_output_audio=True,
    sentences=[
            '송은경 기자 = "뉴스룸" 앵커석에서 물러난 손석희 JTBC 대표이사 사장이 11일 "조국 정국에서 저널리즘 목적에서 벗어나지 않으려고 했다"고 밝혔다.',
            '손 사장은 이날 새벽 자신의 팬카페에 올린 글에서 "세월호와 촛불, 미투, 조국 정국까지 나로서는 그동안 주장해왔던 저널리즘의 두 가지 목적, 인본주의와 민주주의에서 벗어나지 않으려 했는데 평가는 엇갈리게 마련이다"라고 적었다.',
            'JTBC "뉴스룸"은 지난해 조국 전 법무부 장관 이슈가 불거졌을 당시 조 전 장관 지지자들로부터 "편파방송"이라는 원성을 샀다.',
            '9월 서초동 검찰개혁 촛불집회를 생중계하던 "뉴스룸" 화면에는 "돌아오라 손석희"라는 팻말이 등장하기도 했다.',
            '"뉴스룸"을 떠난 손 사장은 자신의 거취에 대해서도 언급했다.',
            '그는 "직책(대표이사 사장)에 따른 일들은 계속하고 있지만, 나 같은 방송장이는 방송을 떠나면 사실은 은퇴한 것이나 마찬가지"라며 "그에 따른 거취를 어떻게 할 것인가는 제가 풀어야 할 과제이기도 하다"라고 말했다.',
            '손 사장은 지난 2일 "뉴스룸" 신년 토론 진행을 끝으로 6년 4개월 만에 주중 앵커 자리에서 물러났다.'
    ]
    # sentences=[
    #     '옛날 어느 마을에 한 총각이 살았습니다.',
    #     '사람들이 그 총각을 밥벌레 장군이라고 놀렸습니다.',
    #     '왜냐하면, 매일 밥만 먹고 빈둥빈둥 놀기 때문입니다.',
    #     '밥도 한두 그릇 먹는 게 아니라 ',
    #     '가마솥 통째로 먹어 치웁니다.',
    #     '그래서 몸집도 엄청 큽니다.',
    #     '그런데 힘은 어찌나 없는지 밥그릇 하나도 들지 못합니다.',
    #     '밥만 먹고 똥만 싸니 집안 살림이 거덜 나게 생겼습니다.',
    #     '걱정된 부모님은 밥벌레 장군을 불러 놓고 말했습니다.',
    #     '"얘야, 더는 널 먹여 살릴 수가 없구나."',
    #     '"집을 나가서 네 힘으로 살아 보아라"',
    #     '집을 나온 밥벌레 장군은 밥을 얻어먹고 다녔습니다.',
    #     '하루는 깊은 산골의 한 초가집을 지나가고 있었습니다.',
    #     '"여기서 밥을 얻어먹을 수 있으면 좋겠다."',
    #     '담 너머를 기웃거리고 있는데 아낙네가 나오더니 말했습니다.',
    #     '"이리 들어오시지요!"',
    #     '아낙네는 남편이 호랑이한테 잡아 먹혀 세 아들이',
    #     '호랑이를 잡으러 날마다 산에 간다는 이야기를 해주었습니다.',
    # ]

    ### test text
# Eval sentences (if no eval file was specified, these sentences are used for eval)
)
