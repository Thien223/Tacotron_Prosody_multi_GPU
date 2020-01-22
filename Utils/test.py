# 이러한 위기 속에 흥선대원군은 철저한 봉쇄 정책으로 일시적으로 접근을 막았으나, 이후 주위의 열강들과 무력 분쟁을 겪는 등 제국주의와 맞서게 되었다. 1870년대 초반에 일본은 조선에 무력으로 압력을 행사하면서 이미 영향력을 행사하고 있던 청나라와 충돌하였고, 한국을 일본의 영향력 아래에 두려고 하였다. 1894년 일본은 청일전쟁을 일으켜 승리함으로써 조선과 요동반도에 대한 영향력을 강화하고자 하였으나 삼국간섭에 의해 잠시 물러섰다. 1895년 10월 8일, 명성황후가 일본 자객들에게 암살되었다. 1897년 10월 12일, 조선은 대한제국으로 국호를 새롭게 정하였고, 고종은 황제의 자리에 올랐다. 그러나 일본의 부당한 요구와 간섭은 점차 거세지고 있었다. 한국의 곳곳에서는 의병들이 항일 무장 투쟁을 전개하였다. 일본은 러일전쟁에서 승리하여 대한제국에 대한 영향력을 크게 증가시킬 수 있었다. 1905년 일본은 대한제국에게 압력을 행사하여 "을사조약"을 강제로 체결함으로써 대한제국을 보호국으로 만들었고 1910년에는 요적인 합방조약을 체결하였다. 그러나, 이 두 조약 모두 강압에 의한 것이며 법적으로는 무효라고 볼 수 있다. 한국인은 일제강점기 동안 일본제국의 점령에 저항하고자 노력하였다. 1919년에는 곳곳에서 비폭력적인 3. 1 운동이 일어났고, 뒤이어 이러한 독립운동을 총괄하고자 대한민국 임시정부가 설립되어 만주와 중국과 시베리아에서 많은 활동을 하였다. 1920년대에는 서로군정서와 같은 독립군이 일본군과 직접적인 전쟁을 벌였고 봉오동 전투, 청산리 전투와 같은 성과를 거두기도 하였다. 1945년 8월 15일 제2차 세계대전에서 일본이 패망하자 한국은 해방을 맞았다. 해방부터 대한민국과 조선민주주의인민공화국이 성립되는 시기까지를 군정기 또는 해방정국이라 한다. 해방정국에서 남과 북은 극심한 좌우 갈등을 겪었다. 북에서는 조만식과 같은 우익 인사에 대한 탄압이 있었고, 남에서는 여운형과 같은 중도 좌파 정치인이 암살되었다. 국제사회에서는 모스크바 3상회의를 통해 소련과 미국에서 통일 임시정부 수립을 위한 신탁통치를 계획했지만, 한국에서의 극심한 반대와 함께 미소공동위원회의 결렬로 폐기되었다. 1948년 대한민국과 조선민주주의인민공화국의 정부가 별도로 수립되면서 일본군의 무장해제를 위한 군사분계선이었던 38선은 두 개의 국가로 분단되는 기준이 되었다. 분단 이후 양측 간의 긴장이 이어졌고 수시로 국지적인 교전이 있었다. 1950년 6월 25일 조선인민군이 일제히 38선을 넘어 대한민국을 침략하여 한국 전쟁이 발발하였다.

import tensorflow as tf

n = 10000
x = tf.constant(list(range(n)))
c = lambda i, x: i < n
b = lambda i, x: (tf.Print(i + 1, [i]), tf.Print(x + 1, [i], "x:"))
i, out = tf.while_loop(c, b, (0, x))
with tf.Session() as sess:
    #print(sess.run(i))  # prints [0] ... [9999]

    # The following line may increment the counter and x in parallel.
    # The counter thread may get ahead of the other thread, but not the
    # other way around. So you may see things like
    # [9996] x:[9987]
    # meaning that the counter thread is on iteration 9996,
    # while the other thread is on iteration 9987
    print(sess.run(out).shape)





from Utils.AudioProcessing.AudioPreprocess import *



wav = load_wav('D:\locs\projects\locs_projects\Tacotron_Korean\LJSpeech-1.1\wavs\LJ001-0001.wav', hparams.sample_rate)

mel = audio_series_to_mel(hparams,wav)

wav_ = mel_to_audio_serie(mel, hparams)

save_wav(wav,'d:\\test.wav', hparams.sample_rate)
# # # #
# # wav = load_wav('F:\\1_0019.wav',sr=44100)
# # wav = np.load('f:\\speech-audio-00001.npy')
# # import matplotlib.pyplot as plt
# # from Utils.Plot import plot_spectrogram
MEL = np.load('tacotron_output/gta/speech-mel-speech-00000-1-kss.npy')
# #linear = np.load('F:\\speech-linear-00001.npy')
WAV = mel_to_audio_serie(MEL.T,hparams)
import matplotlib.pyplot as plt
plt.plot(WAV[:100])
# #wav = linear_to_audio_serie(linear.T, hparams)
# # mel = audio_series_to_mel(hparams,wav)
# # audio = save_wav(wav, 'f:\\test1.wav', sr=44100)
audio = save_wav(WAV, 'D:\\test29_mel.wav', sr=hparams.sample_rate)
# # plt.plot(wav)
# # #
# #


wav = np.load('D:\\speech-audio-00002.npy')
import matplotlib.pylab as plt
plt.figure()
plt.plot(wav)
plt.show()
plt.close()
audio = save_wav(wav, 'D:\\test2_audio.wav', sr=hparams.sample_rate)


wav = load_wav('D:\\LJ050-0273.wav',hparams.sample_rate)
mel = audio_series_to_mel(hparams,wav)
wav = mel_to_audio_serie(mel, hparams)
audio = save_wav(wav, 'D:\\test3_wav.wav', sr=hparams.sample_rate)







wav = load_wav('F:\\1_0000.wav',hparams.sample_rate)
linear = audio_series_to_linear(hparams, wav)
wav = linear_to_audio_serie(linear, hparams)
audio = save_wav(wav, 'f:\\test1.wav', sr=hparams.sample_rate)


linear = np.load('D:\\speech-linear-01771.npy')
wav = linear_to_audio_serie(linear.T, hparams)
audio = save_wav(wav, 'D:\\test1.wav', sr=hparams.sample_rate)

from Utils.TextProcessing.HangulUtils import *
text =['ii', 'll', 'ss', 'ii', 'pf', 'kk', 'uu', 's0', 'ee', 'k0', 'ii', 'ee', ' ', 't0', 'xx', 'rr', 'vv', ' ', 'c0', 'oo', 's0', 'vv', 'nn', 'ee', 'nn', 'xx', 'nf', ' ', 'k0', 'aa', 'th', 'xx', 'nf', ' ', 'wo', 's0', 'ee', 'xi', ' ', 'c0', 'vv', 'pf', 'kk', 'xx', 'nn', 'ii', ' ', 'p0', 'ii', 'nf', 'p0', 'vv', 'nf', 'h0', 'aa', 'k0', 'ee', ' ', 'ii', 'rr', 'vv', 'nn', 'aa', 'tf', 'tt', 'aa', '.', ' ', 'ii', 'rr', 'vv', 'h0', 'aa', 'nf', ' ', 'wi', 'k0', 'ii', ' ', 's0', 'oo', 'k0', 'ee', ' ', 'h0', 'xx', 'ng', 's0', 'vv', 'nf', 't0', 'qq', 'wv', 'nf', 'k0', 'uu', 'nn', 'xx', 'nf', ' ', 'ch', 'vv', 'll', 'c0', 'vv', 'h0', 'aa', 'nf', ' ', 'p0', 'oo', 'ng', 's0', 'wq', ' ', 'c0', 'vv', 'ng', 'ch', 'qq', 'k0', 'xx', 'rr', 'oo', ' ', 'ii', 'll', 'ss', 'ii', 'c0', 'vv', 'k0', 'xx', 'rr', 'oo', ' ', 'c0', 'vv', 'pf', 'kk', 'xx', 'nn', 'xx', 'll', ' ', 'mm', 'aa', 'k0', 'aa', 'ss', 'xx', 'nn', 'aa', ',', ' ', 'ii', 'h0', 'uu', ' ', 'c0', 'uu', 'wi', 'xi', ' ', 'yv', 'll', 'k0', 'aa', 'ng', 't0', 'xx', 'll', 'k0', 'wa', ' ', 'mm', 'uu', 'rr', 'yv', 'kf', ' ', 'p0', 'uu', 'nf', 'c0', 'qq', 'ng', 'xx', 'll', ' ', 'k0', 'yv', 'ng', 'nn', 'xx', 'nf', ' ', 't0', 'xx', 'ng', ' ', 'c0', 'ee', 'k0', 'uu', 'kf', 'cc', 'uu', 'xi', 'wa', ' ', 'mm', 'aa', 'tf', 'ss', 'vv', 'k0', 'ee', ' ', 't0', 'wo', 'vv', 'tf', 'tt', 'aa', '.', ' ', 'ch', 'vv', 'nf', 'ph', 'aa', 'll', 'p0', 'qq', 'kf', 'ch', 'ii', 'll', 'ss', 'ii', 'mf', 'nn', 'yv', 'nf', 't0', 'qq', ' ', 'ch', 'oo', 'p0', 'aa', 'nn', 'ee', ' ', 'ii', 'll', 'p0', 'oo', 'nn', 'xx', 'nf', ' ', 'c0', 'oo', 's0', 'vv', 'nn', 'ee', ' ', 'mm', 'uu', 'rr', 'yv', 'k0', 'xx', 'rr', 'oo', ' ', 'aa', 'mf', 'nn', 'yv', 'k0', 'xx', 'll', ' ', 'h0', 'qq', 'ng', 's0', 'aa', 'h0', 'aa', 'mm', 'yv', 'nf', 's0', 'vv', ' ', 'ii', 'mm', 'ii', ' ', 'yv', 'ng', 'h0', 'ya', 'ng', 'nn', 'yv', 'k0', 'xx', 'll', ' ', 'h0', 'qq', 'ng', 's0', 'aa', 'h0', 'aa', 'k0', 'oo', ' ', 'ii', 'tf', 'tt', 'vv', 'nf', ' ', 'ch', 'vv', 'ng', 'nn', 'aa', 'rr', 'aa', 'wa', ' ', 'ch', 'uu', 'ng', 't0', 'oo', 'll', 'h0', 'aa', 'yv', 'tf', 'kk', 'oo', ',', ' ', 'h0', 'aa', 'nf', 'k0', 'uu', 'k0', 'xx', 'll', ' ', 'ii', 'll', 'p0', 'oo', 'nn', 'xi', ' ', 'yv', 'ng', 'h0', 'ya', 'ng', 'nn', 'yv', ' ', 'k0', 'aa', 'rr', 'qq', 'ee', ' ', 't0', 'uu', 'rr', 'yv', 'k0', 'oo', ' ', 'h0', 'aa', 'yv', 'tf', 'tt', 'aa', '.', ' ', 'ch', 'vv', 'nf', 'ph', 'aa', 'll', 'p0', 'qq', 'kf', 'kk', 'uu', 's0', 'ii', 'pf', 'ss', 'aa', 'nn', 'yv', 'nf', ' ', 'ii', 'll', 'p0', 'oo', 'nn', 'xx', 'nf', ' ', 'ch', 'vv', 'ng', 'ii', 'll', 'c0', 'vv', 'nf', 'c0', 'qq', 'ng', 'xx', 'll', ' ', 'ii', 'rr', 'xx', 'kh', 'yv', ' ', 's0', 'xx', 'ng', 'nn', 'ii', 'h0', 'aa', 'mm', 'xx', 'rr', 'oo', 'ss', 'vv', ' ', 'c0', 'oo', 's0', 'vv', 'nf', 'k0', 'wa', ' ', 'yo', 't0', 'oo', 'ng', 'p0', 'aa', 'nf', 't0', 'oo', 'ee', ' ', 't0', 'qq', 'h0', 'aa', 'nf', ' ', 'nn', 'yv', 'ng', 'h0', 'ya', 'ng', 'nn', 'yv', 'k0', 'xx', 'll', ' ', 'k0', 'aa', 'ng', 'h0', 'wa', 'h0', 'aa', 'k0', 'oo', 'c0', 'aa', ' ', 'h0', 'aa', 'yv', 'ss', 'xx', 'nn', 'aa', ' ', 's0', 'aa', 'mf', 'kk', 'uu', 'kf', 'kk', 'aa', 'nf', 's0', 'vv', 'p0', 'ee', ' ', 'xi', 'h0', 'qq', ' ', 'c0', 'aa', 'mf', 's0', 'ii', ' ', 'mm', 'uu', 'll', 'rr', 'vv', 's0', 'vv', 'tf', 'tt', 'aa', '.', ' ', 'ch', 'vv', 'nf', 'ph', 'aa', 'll', 'p0', 'qq', 'kf', 'kk', 'uu', 's0', 'ii', 'p0', 'oo', 'nn', 'yv', 'nf', ' ', 'ii', 'll', 'ss', 'ii', 'p0', 'wv', 'll', ' ', 'ph', 'aa', 'rr', 'ii', 'll', ',', ' ', 'mm', 'yv', 'ng', 's0', 'vv', 'ng', 'h0', 'wa', 'ng', 'h0', 'uu', 'k0', 'aa', ' ', 'ii', 'll', 'p0', 'oo', 'nf', ' ', 'c0', 'aa', 'k0', 'qq', 'kf', 'tt', 'xx', 'rr', 'ee', 'k0', 'ee', ' ', 'aa', 'mf', 's0', 'aa', 'll', 't0', 'wo', 'vv', 'tf', 'tt', 'aa', '.', 'ii', 'll', 'ph', 'aa', 'll', '9', '7', 'nn', 'yv', 'nf', ' ', 'ii', 'll', 'ss', 'ii', 'p0', 'wv', 'll', ' ', 'ii', 'll', 'ss', 'ii', 'p0', 'ii', 'ii', 'll', ',', ' ', 'c0', 'oo', 's0', 'vv', 'nn', 'xx', 'nf', ' ', 't0', 'qq', 'h0', 'aa', 'nf', 'c0', 'ee', 'k0', 'uu', 'k0', 'xx', 'rr', 'oo', ' ', 'k0', 'uu', 'kh', 'oo', 'rr', 'xx', 'll', ' ', 's0', 'qq', 'rr', 'oo', 'pf', 'kk', 'ee', ' ', 'c0', 'vv', 'ng', 'h0', 'aa', 'yv', 'tf', 'kk', 'oo', ',', ' ', 'k0', 'oo', 'c0', 'oo', 'ng', 'xx', 'nf', ' ', 'h0', 'wa', 'ng', 'c0', 'ee', 'xi', ' ', 'c0', 'aa', 'rr', 'ii', 'ee', ' ', 'oo', 'll', 'rr', 'aa', 'tf', 'tt', 'aa', '.', ' ', 'k0', 'xx', 'rr', 'vv', 'nn', 'aa', ' ', 'ii', 'll', 'p0', 'oo', 'nn', 'xi', ' ', 'p0', 'uu', 't0', 'aa', 'ng', 'h0', 'aa', 'nf', ' ', 'nn', 'yo', 'k0', 'uu', 'wa', ' ', 'k0', 'aa', 'nf', 's0', 'vv', 'p0', 'xx', 'nf', ' ', 'c0', 'vv', 'mf', 'ch', 'aa', ' ', 'k0', 'vv', 's0', 'ee', 'c0', 'ii', 'k0', 'oo', ' ', 'ii', 'ss', 'vv', 'tf', 'tt', 'aa', '.', ' ', 'h0', 'aa', 'nf', 'k0', 'uu', 'k0', 'xi', ' ', 'k0', 'oo', 'tf', 'kk', 'oo', 's0', 'ee', 's0', 'vv', 'nn', 'xx', 'nf', ' ', 'xi', 'p0', 'yv', 'ng', 't0', 'xx', 'rr', 'ii', ' ', 'h0', 'aa', 'ng', 'ii', 'll', ' ', 'mm', 'uu', 'c0', 'aa', 'ng', ' ', 'th', 'uu', 'c0', 'qq', 'ng', 'xx', 'll', ' ', 'c0', 'vv', 'nf', 'k0', 'qq', 'h0', 'aa', 'yv', 'tf', 'tt', 'aa', '.', 'ii', 'll', 'p0', 'oo', 'nn', 'xx', 'nf', ' ', 'rr', 'vv', 'ii', 'll', 'c0', 'vv', 'nf', 'c0', 'qq', 'ng', 'ee', 's0', 'vv', ' ', 's0', 'xx', 'ng', 'nn', 'ii', 'h0', 'aa', 'yv', ' ', 't0', 'qq', 'h0', 'aa', 'nf', 'c0', 'ee', 'k0', 'uu', 'k0', 'ee', ' ', 't0', 'qq', 'h0', 'aa', 'nf', ' ', 'nn', 'yv', 'ng', 'h0', 'ya', 'ng', 'nn', 'yv', 'k0', 'xx', 'll', ' ', 'kh', 'xx', 'k0', 'ee', ' ', 'c0', 'xx', 'ng', 'k0', 'aa', 's0', 'ii', 'kh', 'ii', 'll', ' ', 's0', 'uu', ' ', 'ii', 'ss', 'vv', 'tf', 'tt', 'aa', '.', ' ', 'ii', 'll', 'ss', 'ii', 'pf', 'kk', 'uu', '0', '5', 'nn', 'yv', 'nf', ' ', 'ii', 'll', 'p0', 'oo', 'nn', 'xx', 'nf', ' ', 't0', 'qq', 'h0', 'aa', 'nf', 'c0', 'ee', 'k0', 'uu', 'k0', 'ee', 'k0', 'ee', ' ', 'aa', 'mf', 'nn', 'yv', 'k0', 'xx', 'll', ' ', 'h0', 'qq', 'ng', 's0', 'aa', 'h0', 'aa', 'yv', ' ', '"', 'xx', 'll', 's0', 'aa', 'c0', 'oo', 'ya', 'kf', '"', 'xx', 'll', ' ', 'k0', 'aa', 'ng', 'c0', 'ee', 'rr', 'oo', ' ', 'ch', 'ee', 'k0', 'yv', 'll', 'h0', 'aa', 'mm', 'xx', 'rr', 'oo', 'ss', 'vv', ' ', 't0', 'qq', 'h0', 'aa', 'nf', 'c0', 'ee', 'k0', 'uu', 'k0', 'xx', 'll', ' ', 'p0', 'oo', 'h0', 'oo', 'k0', 'uu', 'k0', 'xx', 'rr', 'oo', ' ', 'mm', 'aa', 'nf', 't0', 'xx', 'rr', 'vv', 'tf', 'kk', 'oo', ' ', 'ii', 'll', 'ss', 'ii', 'pf', 'kk', 'uu', 'ii', 'll', 'ss', 'ii', 'mf', 'nn', 'yv', 'nn', 'ee', 'nn', 'xx', 'nf', ' ', 'nn', 'yo', 'c0', 'vv', 'k0', 'ii', 'nf', ' ', 'h0', 'aa', 'pf', 'pp', 'aa', 'ng', 'c0', 'oo', 'ya', 'k0', 'xx', 'll', ' ', 'ch', 'ee', 'k0', 'yv', 'll', 'h0', 'aa', 'yv', 'tf', 'tt', 'aa', '.', ' ', 'k0', 'xx', 'rr', 'vv', 'nn', 'aa', ',', ' ', 'ii', ' ', 't0', 'uu', ' ', 'c0', 'oo', 'ya', 'ng', ' ', 'mm', 'oo', 't0', 'uu', ' ', 'k0', 'aa', 'ng', 'aa', 'p0', 'ee', ' ', 'xi', 'h0', 'aa', 'nf', ' ', 'k0', 'vv', 's0', 'ii', 'mm', 'yv', ' ', 'p0', 'vv', 'pf', 'cc', 'vv', 'k0', 'xx', 'rr', 'oo', 'nn', 'xx', 'nf', ' ', 'mm', 'uu', 'h0', 'yo', 'rr', 'aa', 'k0', 'oo', ' ', 'p0', 'oo', 'll', ' ', 's0', 'uu', ' ', 'ii', 'tf', 'tt', 'aa', '.', ' ', 'h0', 'aa', 'nf', 'k0', 'uu', 'k0', 'ii', 'nn', 'xx', 'nf', ' ', 'ii', 'll', 'c0', 'ee', 'k0', 'aa', 'ng', 'c0', 'vv', 'mf', 'k0', 'ii', ' ', 't0', 'oo', 'ng', 'aa', 'nf', ' ', 'ii', 'll', 'p0', 'oo', 'nf', 'c0', 'ee', 'k0', 'uu', 'k0', 'xi', ' ', 'c0', 'vv', 'mf', 'nn', 'yv', 'ng', 'ee', ' ', 'c0', 'vv', 'h0', 'aa', 'ng', 'h0', 'aa', 'k0', 'oo', 'c0', 'aa', ' ', 'nn', 'oo', 'rr', 'yv', 'kh', 'aa', 'yv', 'tf', 'tt', 'aa', '.', ' ', 'ii', 'll', 'ss', 'ii', 'pf', 'kk', 'uu', 'ii', 'll', 'ss', 'ii', 'pf', 'kk', 'uu', 'nn', 'yv', 'nn', 'ee', 'nn', 'xx', 'nf', ' ', 'k0', 'oo', 'tf', 'kk', 'oo', 's0', 'ee', 's0', 'vv', ' ', 'p0', 'ii', 'ph', 'oo', 'ng', 'nn', 'yv', 'kf', 'cc', 'vv', 'k0', 'ii', 'nf', ' ', 's0', 'aa', 'mf', '.', 'ii', 'll', ' ', 'uu', 'nf', 't0', 'oo', 'ng', 'ii', ' ', 'ii', 'rr', 'vv', 'nn', 'aa', 'tf', 'kk', 'oo', ',', ' ', 't0', 'wi', 'ii', 'vv', ' ', 'ii', 'rr', 'vv', 'h0', 'aa', 'nf', ' ', 't0', 'oo', 'ng', 'nn', 'ii', 'p0', 'uu', 'nf', 't0', 'oo', 'ng', 'xx', 'll', ' ', 'ch', 'oo', 'ng', 'k0', 'wa', 'll', 'h0', 'aa', 'k0', 'oo', 'c0', 'aa', ' ', 't0', 'qq', 'h0', 'aa', 'nf', 'mm', 'ii', 'nf', 'k0', 'uu', ' ', 'k0', 'ii', 'mf', 's0', 'ii', 'c0', 'vv', 'ng', 'p0', 'uu', 'k0', 'aa', ' ', 's0', 'vv', 'll', 'rr', 'ii', 'pf', 'tt', 'wo', 'vv', ' ', 'mm', 'aa', 'nf', 'c0', 'uu', 'wa', ' ', 'c0', 'uu', 'ng', 'k0', 'uu', 'kf', 'kk', 'wa', ' ', 's0', 'ii', 'p0', 'ee', 'rr', 'ii', 'aa', 'ee', 's0', 'vv', ' ', 'mm', 'aa', 'nn', 'xx', 'nf', ' ', 'h0', 'wa', 'll', 't0', 'oo', 'ng', 'xx', 'll', ' ', 'h0', 'aa', 'yv', 'tf', 'tt', 'aa', '.', ' ', 'ii', 'll', 'ss', 'ii', 'pf', 'kk', 'uu', 'ii', '0', 'nn', 'yv', 'nf', 't0', 'qq', 'ee', 'nn', 'xx', 'nf', ' ', 's0', 'vv', 'rr', 'oo', 'k0', 'uu', 'nf', 'c0', 'vv', 'ng', 's0', 'vv', 'wa', ' ', 'k0', 'aa', 'th', 'xx', 'nf', ' ', 't0', 'oo', 'ng', 'nn', 'ii', 'pf', 'kk', 'uu', 'nn', 'ii', ' ', 'ii', 'll', 'p0', 'oo', 'nf', 'k0', 'uu', 'nf', 'k0', 'wa', ' ', 'c0', 'ii', 'kf', 'cc', 'vv', 'pf', 'cc', 'vv', 'k0', 'ii', 'nf', ' ', 'c0', 'vv', 'nf', 'c0', 'qq', 'ng', 'xx', 'll', ' ', 'p0', 'vv', 'll', 'rr', 'yv', 'tf', 'kk', 'oo', ' ', 'p0', 'oo', 'ng', 'oo', 't0', 'oo', 'ng', ' ', 'c0', 'vv', 'nf', 'th', 'uu', ',', ' ', 'ch', 'vv', 'ng', 's0', 'aa', 'll', 'rr', 'ii', ' ', 'c0', 'vv', 'nf', 'th', 'uu', 'wa', ' ', 'k0', 'aa', 'th', 'xx', 'nf', ' ', 's0', 'vv', 'ng', 'k0', 'wa', 'rr', 'xx', 'll', ' ', 'k0', 'vv', 't0', 'uu', 'k0', 'ii', 't0', 'oo', ' ', 'h0', 'aa', 'yv', 'tf', 'tt', 'aa', '.', 'ii', 'll', 'ss', 'ii', 'pf', 'kk', 'uu', '4', '5', 'nn', 'yv', 'nf', ' ', 'ph', 'aa', 'rr', 'wv', 'll', ' ', 'ii', 'll', '5', 'ii', 'll', ' ', 'c0', 'ee', 'ii', 'ch', 'aa', ' ', 's0', 'ee', 'k0', 'ye', 't0', 'qq', 'c0', 'vv', 'nn', 'ee', 's0', 'vv', ' ', 'ii', 'll', 'p0', 'oo', 'nn', 'ii', ' ', 'ph', 'qq', 'mm', 'aa', 'ng', 'h0', 'aa', 'c0', 'aa', ' ', 'h0', 'aa', 'nf', 'k0', 'uu', 'k0', 'xx', 'nf', ' ', 'h0', 'qq', 'p0', 'aa', 'ng', 'xx', 'll', ' ', 'mm', 'aa', 'c0', 'aa', 'tf', 'tt', 'aa', '.', ' ', 'h0', 'qq', 'p0', 'aa', 'ng', 'p0', 'uu', 'th', 'vv', ' ', 't0', 'qq', 'h0', 'aa', 'nf', 'mm', 'ii', 'nf', 'k0', 'uu', 'kf', 'kk', 'wa', ' ', 'c0', 'oo', 's0', 'vv', 'nf', 'mm', 'ii', 'nf', 'c0', 'uu', 'c0', 'uu', 'xi', 'ii', 'nf', 'mm', 'ii', 'nf', 'k0', 'oo', 'ng', 'h0', 'wa', 'k0', 'uu', 'k0', 'ii', ' ', 's0', 'vv', 'ng', 'nn', 'ii', 'pf', 'tt', 'wo', 'nn', 'xx', 'nf', ' ', 's0', 'ii', 'k0', 'ii', 'kk', 'aa', 'c0', 'ii', 'rr', 'xx', 'll', ' ', 'k0', 'uu', 'nf', 'c0', 'vv', 'ng', 'k0', 'ii', ' ', 'tt', 'oo', 'nn', 'xx', 'nf', ' ', 'h0', 'qq', 'p0', 'aa', 'ng', 'c0', 'vv', 'ng', 'k0', 'uu', 'k0', 'ii', 'rr', 'aa', ' ', 'h0', 'aa', 'nf', 't0', 'aa', '.', ' ', 'h0', 'qq', 'p0', 'aa', 'ng', 'c0', 'vv', 'ng', 'k0', 'uu', 'k0', 'ee', 's0', 'vv', ' ', 'nn', 'aa', 'mf', 'k0', 'wa', ' ', 'p0', 'uu', 'k0', 'xx', 'nf', ' ', 'k0', 'xx', 'kf', 'ss', 'ii', 'mf', 'h0', 'aa', 'nf', ' ', 'c0', 'wa', 'uu', ' ', 'k0', 'aa', 'll', 'tt', 'xx', 'ng', 'xx', 'll', ' ', 'k0', 'yv', 'kk', 'vv', 'tf', 'tt', 'aa', '.', ' ', 'p0', 'uu', 'k0', 'ee', 's0', 'vv', 'nn', 'xx', 'nf', ' ', 'c0', 'oo', 'mm', 'aa', 'nf', 's0', 'ii', 'kf', 'kk', 'wa', ' ', 'k0', 'aa', 'th', 'xx', 'nf', ' ', 'uu', 'ii', ' ', 'k0', 'ii', 'nf', 's0', 'aa', 'ee', ' ', 't0', 'qq', 'h0', 'aa', 'nf', ' ', 'th', 'aa', 'nn', 'aa', 'p0', 'ii', ' ', 'ii', 'ss', 'vv', 'tf', 'kk', 'oo', ',', ' ', 'nn', 'aa', 'mm', 'ee', 's0', 'vv', 'nn', 'xx', 'nf', ' ', 'nn', 'yv', 'uu', 'nf', 'h0', 'yv', 'ng', 'k0', 'wa', ' ', 'k0', 'aa', 'th', 'xx', 'nf', ' ', 'c0', 'uu', 'ng', 't0', 'oo', ' ', 'c0', 'wa', 'ph', 'aa', ' ', 'c0', 'vv', 'ng', 'ch', 'ii', 'ii', 'nn', 'ii', ' ', 'aa', 'mf', 's0', 'aa', 'll', 't0', 'wo', 'vv', 'tf', 'tt', 'aa', '.', ' ', 'k0', 'uu', 'kf', 'cc', 'ee', 's0', 'aa', 'h0', 'wo', 'ee', 's0', 'vv', 'nn', 'xx', 'nf', ' ', 'mm', 'oo', 's0', 'xx', 'kh', 'xx', 'p0', 'aa', ' ', 's0', 'aa', 'mf', 's0', 'aa', 'ng', 'h0', 'wo', 'xi', 'rr', 'xx', 'll', ' ', 'th', 'oo', 'ng', 'h0', 'qq', ' ', 's0', 'oo', 'rr', 'yv', 'nf', 'k0', 'wa', ' ', 'mm', 'ii', 'k0', 'uu', 'k0', 'ee', 's0', 'vv', ' ', 'th', 'oo', 'ng', 'ii', 'll', ' ', 'ii', 'mf', 's0', 'ii', 'c0', 'vv', 'ng', 'p0', 'uu', ' ', 's0', 'uu', 'rr', 'ii', 'p0', 'xx', 'll', ' ', 'wi', 'h0', 'aa', 'nf', ' ', 's0', 'ii', 'nf', 'th', 'aa', 'kf', 'th', 'oo', 'ng', 'ch', 'ii', 'rr', 'xx', 'll', ' ', 'k0', 'ye', 'h0', 'wo', 'kh', 'qq', 'tf', 'cc', 'ii', 'mm', 'aa', 'nf', ',', ' ', 'h0', 'aa', 'nf', 'k0', 'uu', 'k0', 'ee', 's0', 'vv', 'xi', ' ', 'k0', 'xx', 'kf', 'ss', 'ii', 'mf', 'h0', 'aa', 'nf', ' ', 'p0', 'aa', 'nf', 't0', 'qq', 'wa', ' ', 'h0', 'aa', 'mf', 'kk', 'ee', ' ', 'mm', 'ii', 's0', 'oo', 'k0', 'oo', 'ng', 't0', 'oo', 'ng', 'wi', 'wv', 'nf', 'h0', 'wo', 'xi', ' ', 'k0', 'yv', 'll', 'rr', 'yv', 'll', 'rr', 'oo', ' ', 'ph', 'ye', 'k0', 'ii', 't0', 'wo', 'vv', 'tf', 'tt', 'aa', '.', ' ', 'ii', 'll', 'ss', 'ii', 'pf', 'kk', 'uu', '4', 'ph', 'aa', 'll', 'rr', 'yv', 'nf', ' ', 't0', 'qq', 'h0', 'aa', 'nf', 'mm', 'ii', 'nf', 'k0', 'uu', 'kf', 'kk', 'wa', ' ', 'c0', 'oo', 's0', 'vv', 'nf', 'mm', 'ii', 'nf', 'c0', 'uu', 'c0', 'uu', 'xi', 'ii', 'nf', 'mm', 'ii', 'nf', 'k0', 'oo', 'ng', 'h0', 'wa', 'k0', 'uu', 'k0', 'xi', ' ', 'c0', 'vv', 'ng', 'p0', 'uu', 'k0', 'aa', ' ', 'p0', 'yv', 'll', 't0', 'oo', 'rr', 'oo', ' ', 's0', 'uu', 'rr', 'ii', 'pf', 'tt', 'wo', 'mm', 'yv', 'nf', 's0', 'vv', ' ', 'ii', 'll', 'p0', 'oo', 'nf', 'k0', 'uu', 'nn', 'xi', ' ', 'mm', 'uu', 'c0', 'aa', 'ng', 'h0', 'qq', 'c0', 'ee', 'rr', 'xx', 'll', ' ', 'wi', 'h0', 'aa', 'nf', ' ', 'k0', 'uu', 'nf', 's0', 'aa', 'p0', 'uu', 'nf', 'k0', 'ye', 's0', 'vv', 'nn', 'ii', 'vv', 'tf', 'tt', 'vv', 'nf', ' ', 's0', 'aa', 'mf', 'ph', 'aa', 'll', 's0', 'vv', 'nn', 'xx', 'nf', ' ', 't0', 'uu', ' ', 'k0', 'qq', 'xi', ' ', 'k0', 'uu', 'kf', 'kk', 'aa', 'rr', 'oo', ' ', 'p0', 'uu', 'nf', 't0', 'aa', 'nf', 't0', 'wo', 'nn', 'xx', 'nf', ' ', 'k0', 'ii', 'c0', 'uu', 'nn', 'ii', ' ', 't0', 'wo', 'vv', 'tf', 'tt', 'aa', '.', ' ', 'p0', 'uu', 'nf', 't0', 'aa', 'nf', ' ', 'ii', 'h0', 'uu', ' ', 'ya', 'ng', 'ch', 'xx', 'kf', ' ', 'k0', 'aa', 'nn', 'xi', ' ', 'k0', 'ii', 'nf', 'c0', 'aa', 'ng', 'ii', ' ', 'ii', 'vv', 'c0', 'yv', 'tf', 'kk', 'oo', ' ', 's0', 'uu', 's0', 'ii', 'rr', 'oo', ' ', 'k0', 'uu', 'kf', 'cc', 'ii', 'c0', 'vv', 'k0', 'ii', 'nf', ' ', 'k0', 'yo', 'c0', 'vv', 'nn', 'ii', ' ', 'ii', 'ss', 'vv', 'tf', 'tt', 'aa', '.', ' ', 'ii', 'll', 'ss', 'ii', 'pf', 'kk', 'uu', '5', '0', 'nn', 'yv', 'nf', ' ', 'nn', 'yu', 'k0', 'wv', 'll', ' ', 'ii', '5', 'ii', 'll', ' ', 'c0', 'oo', 's0', 'vv', 'nn', 'ii', 'nf', 'mm', 'ii', 'nf', 'k0', 'uu', 'nn', 'ii', ' ', 'ii', 'll', 'c0', 'ee', 'h0', 'ii', ' ', 's0', 'aa', 'mf', 'ph', 'aa', 'll', 's0', 'vv', 'nn', 'xx', 'll', ' ', 'rr', 'vv', 'mm', 'vv', ' ', 't0', 'qq', 'h0', 'aa', 'nf', 'mm', 'ii', 'nf', 'k0', 'uu', 'k0', 'xx', 'll', ' ', 'ch', 'ii', 'mf', 'nn', 'ya', 'kh', 'aa', 'yv', ' ', 'h0', 'aa', 'nf', 'k0', 'uu', 'kf', ' ', 'c0', 'vv', 'nf', 'c0', 'qq', 'ng', 'ii', ' ', 'p0', 'aa', 'll', 'p0', 'aa', 'll', 'h0', 'aa', 'yv', 'tf', 'tt', 'aa', '.', '␃']
import re
text = '19세기에 들어 조선에는 같은 외세의 접근이 빈번하게 일어났다. 이러한 위기 속에 흥선대원군은 철저한 봉쇄 정책으로 일시적으로 접근을 막았으나, 이후 주위의 열강들과 무력 분쟁을 겪는 등 제국주의와 맞서게 되었다. 1870년대 초반에 일본은 조선에 무력으로 압력을 행사하면서 이미 영향력을 행사하고 있던 청나라와 충돌하였고, 한국을 일본의 영향력 아래에 두려고 하였다. 1894년 일본은 청일전쟁을 일으켜 승리함으로써 조선과 요동반도에 대한 영향력을 강화하고자 하였으나 삼국간섭에 의해 잠시 물러섰다. 1895년 10월 8일, 명성황후가 일본 자객들에게 암살되었다. 1897년 10월 12일, 조선은 대한제국으로 국호를 새롭게 정하였고, 고종은 황제의 자리에 올랐다. 그러나 일본의 부당한 요구와 간섭은 점차 거세지고 있었다. 한국의 곳곳에서는 의병들이 항일 무장 투쟁을 전개하였다. 일본은 러일전쟁에서 승리하여 대한제국에 대한 영향력을 크게 증가시킬 수 있었다. 1905년 일본은 대한제국에게 압력을 행사하여 "을사조약"을 강제로 체결함으로써 대한제국을 보호국으로 만들었고 1910년에는 요적인 합방조약을 체결하였다. 그러나, 이 두 조약 모두 강압에 의한 것이며 법적으로는 무효라고 볼 수 있다. 한국인은 일제강점기 동안 일본제국의 점령에 저항하고자 노력하였다. 1919년에는 곳곳에서 비폭력적인 3. 1 운동이 일어났고, 뒤이어 이러한 독립운동을 총괄하고자 대한민국 임시정부가 설립되어 만주와 중국과 시베리아에서 많은 활동을 하였다. 1920년대에는 서로군정서와 같은 독립군이 일본군과 직접적인 전쟁을 벌였고 봉오동 전투, 청산리 전투와 같은 성과를 거두기도 하였다. 1945년 8월 15일 제2차 세계대전에서 일본이 패망하자 한국은 해방을 맞았다. 해방부터 대한민국과 조선민주주의인민공화국이 성립되는 시기까지를 군정기 또는 해방정국이라 한다. 해방정국에서 남과 북은 극심한 좌우 갈등을 겪었다. 북에서는 조만식과 같은 우익 인사에 대한 탄압이 있었고, 남에서는 여운형과 같은 중도 좌파 정치인이 암살되었다. 국제사회에서는 모스크바 3상회의를 통해 소련과 미국에서 통일 임시정부 수립을 위한 신탁통치를 계획했지만, 한국에서의 극심한 반대와 함께 미소공동위원회의 결렬로 폐기되었다. 1948년 대한민국과 조선민주주의인민공화국의 정부가 별도로 수립되면서 일본군의 무장해제를 위한 군사분계선이었던 38선은 두 개의 국가로 분단되는 기준이 되었다. 분단 이후 양측 간의 긴장이 이어졌고 수시로 국지적인 교전이 있었다. 1950년 6월 25일 조선인민군이 일제히 38선을 넘어 대한민국을 침략하여 한국 전쟁이 발발하였다.'

text = '19세기에 들어 조선에는 같은 외세의 접근이 빈번하게 일어났다. 이러한 위기 속에 흥선대원군은 철저한 봉쇄 정책으로 일시적으로 접근을 막았으나, 이후 주위의 열강들과 무력 분쟁을 겪는 등 제국주의와 맞서게 되었다. 1870년대 초반에 일본은 조선에 무력으로 압력을 행사하면서 이미 영향력을 행사하고 있던 청나라와 충돌하였고, 한국을 일본의 영향력 아래에 두려고 하였다. 1894년 일본은 청일전쟁을 일으켜 승리함으로써 조선과 요동반도에 대한 영향력을 강화하고자 하였으나 삼국간섭에 의해 잠시 물러섰다. 1895년 10월 8일, 명성황후가 일본 자객들에게 암살되었다. 1897년 10월 12일, 조선은 대한제국으로 국호를 새롭게 정하였고, 고종은 황제의 자리에 올랐다.'
print(number_to_hangul(text))
numbers = re.findall(r"[+-]?\d+(?:\.\d+)?", text)

i = 0
for id,x in enumerate(text):
    print(x)
    print(id)
    #print(hangul_to_ids[x])
     #i = i + 1


from Utils.AudioProcessing.trimming import *
output_dir=None
input_wav_files = 'D:\locs\data\\audio_data\\20대 남성\mv19\\6minutes\\'
resample_wav(input_wav_files=input_wav_files,sample_rate=hparams.sample_rate)
input_audio_files='D:\\locs\data\\audio data\\zeroth_korean_male_20_6minutes\\'
convert_to_wav(input_wav_files)
input_wav_files = 'D:\\locs\data\\audio_data\\zeroth_korean_male_20_6minutes\\converted\\wavs_16000Hz\\'
trimming(input_wav_files)

