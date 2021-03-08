import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

class CMVN(torch.jit.ScriptModule):

    __constants__ = ["mode", "dim", "eps"]

    def __init__(self, mode="global", dim=2, eps=1e-10):
        # `torchaudio.load()` loads audio with shape [channel, feature_dim, time]
        # so perform normalization on dim=2 by default
        super(CMVN, self).__init__()

        if mode != "global":
            raise NotImplementedError(
                "Only support global mean variance normalization.")

        self.mode = mode
        self.dim = dim
        self.eps = eps

    @torch.jit.script_method
    def forward(self, x):
        if self.mode == "global":
            return (x - x.mean(self.dim, keepdim=True)) / (self.eps + x.std(self.dim, keepdim=True))

    def extra_repr(self):
        return "mode={}, dim={}, eps={}".format(self.mode, self.dim, self.eps)


class Delta(torch.jit.ScriptModule):

    __constants__ = ["order", "window_size", "padding"]

    def __init__(self, order=1, window_size=2):
        # Reference:
        # https://kaldi-asr.org/doc/feature-functions_8cc_source.html
        # https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/layers/common_audio.py
        super(Delta, self).__init__()

        self.order = order
        self.window_size = window_size

        filters = self._create_filters(order, window_size)
        self.register_buffer("filters", filters)
        self.padding = (0, (filters.shape[-1] - 1) // 2)

    @torch.jit.script_method
    def forward(self, x):
        # Unsqueeze batch dim
        x = x.unsqueeze(0)
        return F.conv2d(x, weight=self.filters, padding=self.padding)[0]

    # TODO(WindQAQ): find more elegant way to create `scales`
    def _create_filters(self, order, window_size):
        scales = [[1.0]]
        for i in range(1, order + 1):
            prev_offset = (len(scales[i-1]) - 1) // 2
            curr_offset = prev_offset + window_size

            curr = [0] * (len(scales[i-1]) + 2 * window_size)
            normalizer = 0.0
            for j in range(-window_size, window_size + 1):
                normalizer += j * j
                for k in range(-prev_offset, prev_offset + 1):
                    curr[j+k+curr_offset] += (j * scales[i-1][k+prev_offset])
            curr = [x / normalizer for x in curr]
            scales.append(curr)

        max_len = len(scales[-1])
        for i, scale in enumerate(scales[:-1]):
            padding = (max_len - len(scale)) // 2
            scales[i] = [0] * padding + scale + [0] * padding

        return torch.tensor(scales).unsqueeze(1).unsqueeze(1)

    def extra_repr(self):
        return "order={}, window_size={}".format(self.order, self.window_size)


class Postprocess(torch.jit.ScriptModule):
    @torch.jit.script_method
    def forward(self, x):
        # [channel, feature_dim, time] -> [time, channel, feature_dim]
        x = x.permute(2, 0, 1)
        # [time, channel, feature_dim] -> [time, feature_dim * channel]
        return x.reshape(x.size(0), -1).detach()



# TODO(Windqaq): make this scriptable
class ExtractAudioFeature(nn.Module):
    def __init__(self, mode="fbank", num_mel_bins=40, **kwargs):
        super(ExtractAudioFeature, self).__init__()
        self.mode = mode
        self.extract_fn = torchaudio.compliance.kaldi.fbank if mode == "fbank" else torchaudio.compliance.kaldi.spectrogram
        self.num_mel_bins = num_mel_bins
        self.kwargs = kwargs

    def forward(self, filepath):
        waveform, sample_rate = torchaudio.load(filepath)

        #log mel spec
        y = torchaudio.transforms.MelSpectrogram(n_mels=80)(waveform)
        return y.log2().detach()
        '''
        y = self.extract_fn(waveform,
                            num_mel_bins=self.num_mel_bins,
                            channel=-1,
                            sample_frequency=sample_rate,
                            **self.kwargs)

        return y.transpose(0, 1).unsqueeze(0).detach()
        '''

    def extra_repr(self):
        return "mode={}, num_mel_bins={}".format(self.mode, self.num_mel_bins)

def freqMask(x,v,num):
    _num = int(num)
    for _ in range(_num):
        freq_percentage = (v - 0.0) * torch.rand(1) + 0.0
        all_freqs_num = x.size(1)
        num_freqs_to_mask = int(freq_percentage * all_freqs_num)
        f0 = (all_freqs_num - num_freqs_to_mask- 0.0) * torch.rand(1) + 0.0 
        f0 = int(f0)
        x[: , f0:f0 + num_freqs_to_mask] = 0
    return x

def timeMask(x,v,num):
    _num = int(num)
    for _ in range(_num):
        time_percentage = (v - 0.0) * torch.rand(1) + 0.0
        all_frames_num  = x.size(0)
        num_frames_to_mask = int(time_percentage * all_frames_num)
        t0 = (all_frames_num - num_frames_to_mask- 0.0) * torch.rand(1) + 0.0 
        t0 = int(t0)
        x[t0:t0 + num_frames_to_mask , :] = 0
    return x

def loudnessCtl(x,v):
    all_frames_num  = x.size(0)
    loudness_level = (v - 0.5) * torch.rand(1) + 0.5
    min_value = torch.min(x)
    x = x - min_value
    time_start = (0.8 - 0.2) * torch.rand(1) + 0.2
    time_end = (0.8 - time_start) * torch.rand(1) + time_start
    time_start = int (time_start * all_frames_num)
    time_end = int(time_end * all_frames_num)
    x[time_start:time_end,:] = x[time_start:time_end,:] * loudness_level 
    x = x + min_value
    return x

def time_length_ctl(x,v):
    w ,h = x.shape
    time_warp_value = int (w * v)
    x = x.reshape(1,1,w, h)
    resized_img = F.interpolate(x, [w - time_warp_value,h]) 
    resized_img = resized_img[0][0]
    out = F.pad(input=resized_img, pad=(0, 0, 0, time_warp_value), mode='constant', value=0.0)
    return out

def freq_warp(x,v):
    w ,h = x.shape
    freq_warp_value = int (h * v)
    x = x.reshape(1,1,w, h)
    resized_img = F.interpolate(x, [w,h-freq_warp_value]) 
    resized_img = resized_img[0][0]
    out = F.pad(input=resized_img, pad=(0, freq_warp_value, 0, 0), mode='constant', value=0.0)
    return out

class Augment(torch.jit.ScriptModule):

    def __init__(self, policies ,aug_list):
        super(Augment, self).__init__()
        self.policies = policies
        self.aug_list= aug_list

    

    @torch.jit.script_method
    def forward(self, x):
        
        pos = int( len(self.policies) * torch.rand(1) ) 
        policy = self.policies[pos]

        for _ in range(1):
            for name, pr, level in policy:
                if torch.rand(1) > float(pr):
                    continue
                for aug in self.aug_list:
                    augment_fn, low, high = aug
                    if (augment_fn== name):
                        _level = torch.tensor(float(level))
                        _high = torch.tensor(float(high))
                        _low = torch.tensor(float(low))

                        v = _level * (_high - _low) + _low

                        if (augment_fn == 'freqMask_one'): 
                            x = freqMask(x,v,torch.tensor(1))
                        elif (augment_fn == 'timeMask_one'): 
                            x = timeMask(x,v,torch.tensor(1))
                        elif (augment_fn == 'freqMask_two'): 
                            x = freqMask(x,v,torch.tensor(2))
                        elif (augment_fn == 'timeMask_two'): 
                            x = timeMask(x,v,torch.tensor(2))
                        elif (augment_fn == 'loudnessCtl'): 
                            x = loudnessCtl(x,v)
                        elif (augment_fn == 'time_length_ctl'): 
                            x = time_length_ctl(x,v)
                        elif (augment_fn == 'freq_warp'): 
                            x = freq_warp(x,v)
                        break
        return x


def create_transform(audio_config): # create aug
    feat_type = audio_config.pop("feat_type")
    feat_dim = audio_config.pop("feat_dim")

    delta_order = audio_config.pop("delta_order", 0)
    delta_window_size = audio_config.pop("delta_window_size", 2)
    apply_cmvn = audio_config.pop("apply_cmvn")
    augmentation = audio_config.pop("augmentation")

    transforms = [ExtractAudioFeature(feat_type, feat_dim, **audio_config)]

    if delta_order >= 1:
        transforms.append(Delta(delta_order, delta_window_size))

    if apply_cmvn:
        transforms.append(CMVN())

    transforms.append(Postprocess())

    if augmentation == 'test_aug':

        aug_list = [
            ['freqMask_one', '0.0', '0.15'],  # 0
            ['timeMask_one', '0.0', '0.20'],  # 1
            ['freqMask_two', '0.0', '0.15'],  # 0
            ['timeMask_two', '0.0', '0.20'],  # 1
            ['loudnessCtl',  '0.5', '2.00'],  # 1
            ['time_length_ctl',  '0.0', '0.2'],  # 1
            ['freq_warp',  '0.0', '0.1'],  # 1
        ]

        policies = [
        #[['freqMask_one', '1.0', '1.0'],['timeMask_one', '0.0', '1.0']],
        #[['timeMask_one', '1.0', '1.0'],['freqMask_one', '0.0', '1.0']],
        #[['freqMask_two', '1.0', '1.0'],['timeMask_one', '0.0', '1.0']],
        #[['timeMask_two', '1.0', '1.0'],['freqMask_one', '0.0', '1.0']],
        #[['loudnessCtl', '1.0', '1.0'],['freqMask_one', '0.0', '1.0']],
        #[['time_length_ctl', '1.0', '1.0'],['freqMask_one', '0.0', '1.0']],
        [['freq_warp', '1.0', '1.0'],['freqMask_one', '0.0', '1.0']],
        ]

        transforms.append( Augment(policies, aug_list) )

    return nn.Sequential(*transforms), feat_dim * (delta_order + 1)
