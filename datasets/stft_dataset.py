# Copyright (C) 2024 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later


from typing import Dict, List, Union

import numpy as np
import soundfile as sf
import torch

from utils.audio_utils import do_stft, rotate_channels


class STFTDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        path_list: List,
        training: bool,
        stft_conf: Dict,
        sr: int = 8000,
        chunk_size: int = None,
        chunking_strategy: str = "random",
        ref_channel: int = 0,
        channel_idx: Union[int, List[int]] = None,
        running_eras: bool = False,
        normalization: bool = False,
    ):
        """Dataset class.

        Parameters
        ----------
        path_list: List[Dict]
            List of the metadata of each audio data.
        training: bool
            Whether or not this dataset is initialized for training or not.
        stft_conf: Dict
            STFT configuration.
        sr: int
            Sampling rate.
        chunk_size: int or float
            Input length in seconds during training.
        chunking_strategy: str
            How to make a training chunk.
            Choices are random, partial_overlap, or full_overlap.
            NOTE: this is considered only for SMS-WSJ.
        ref_channel: int
            Reference channel index.
        channel_idx: int or List[int]
            Channel index (indices) of the input data.
            To return multi-channel sources, specify the list.
        running_eras: bool
            Whether the training algorithm is ERAS or not.
            ERAS needs a batch which includes both left and right channels in the same mini-batch
            for the ICC loss (i.e., batch: [data1_L, data1_R, ..., dataN_L, dataN_R]).
        normalization: bool
            Whether to apply the variance normalization.
        """

        self.training = training
        self.running_eras = running_eras and self.training
        if self.running_eras:
            print(
                f"Use {channel_idx}-th channel(s) as reference microphone(s)",
                flush=True,
            )

        self.path_list = path_list
        self.sr = sr

        self.chunk_size = int(chunk_size * sr) if self.training else None

        self.ref_channel = ref_channel

        self.stft_conf = stft_conf
        self.channel_idx = channel_idx
        self.normalization = normalization

        assert chunking_strategy in ["random", "partial_overlap", "full_overlap"]
        self.chunking_stragegy = chunking_strategy

    def __len__(self):
        return len(self.path_list)

    def __getitem__(self, index):
        y_mix, y_srcs = self._read_audio(index)
        y_mix = torch.from_numpy(y_mix)
        if isinstance(y_srcs, dict):
            y_srcs = {tag: torch.from_numpy(y_srcs[tag]) for tag in y_srcs.keys()}
        elif y_srcs is not None:
            y_srcs = torch.from_numpy(y_srcs)
        mix_stft = self._stft(y_mix.T)  # (frame, freq, n_chan)

        srcs = dict(y_mix=y_mix, y_srcs=y_srcs)
        if self.running_eras:
            mix_stft, srcs = self._rotate_and_stack_channels_stft(mix_stft, srcs)
        srcs["y_mix_stft"] = mix_stft
        return mix_stft, srcs

    def _read_audio(self, index, start_samp=0, end_samp=None):
        path_dict = self.path_list[index % len(self.path_list)]
        mix_path = path_dict["mix"]
        n_frames = sf.info(mix_path).frames

        # chunking, for SMS-WSJ
        if self.chunk_size is None or n_frames <= self.chunk_size:
            if self.chunking_stragegy == "full_overlap":
                offset = max(path_dict["offset"])  # start of shorter utterance
                min_length = min(path_dict["num_samples"])  # length of shorter utterance
                start_samp, end_samp = offset, min_length + offset
            else:
                start_samp = 0
                end_samp = None
        else:  # n_frames > self.frame_size:
            # in some partially-overlapped data we need to find fully-overlapped segment
            # now we assume number of sources is two
            if self.chunking_stragegy == "full_overlap":
                assert "num_samples" in path_dict and "offset" in path_dict
                offset = max(path_dict["offset"])  # start of shorter utterance
                min_length = min(path_dict["num_samples"])  # length of shorter utterance

                if min_length <= self.chunk_size:
                    start_samp, end_samp = offset, min_length + offset
                else:
                    start_samp = np.random.randint(offset, min_length + offset - self.chunk_size)
                    end_samp = start_samp + self.chunk_size

            elif self.chunking_stragegy == "partial_overlap":
                start_samp, end_samp = self._random_start_and_end(path_dict, min_overlap=self.chunk_size // 4)

            elif self.chunking_stragegy == "random":
                start_samp = np.random.randint(0, n_frames - self.chunk_size)
                end_samp = start_samp + self.chunk_size

            else:
                raise NotImplementedError(self.chunking_stragegy)

        y_mix = self._read_wav(mix_path, start_samp, end_samp)
        y_srcs = {"reverb": []}
        for tag, src_path in path_dict["srcs"].items():
            if "reverb" in tag:
                y_srcs["reverb"].append(self._read_wav(src_path, start_samp, end_samp))
        assert len(y_srcs["reverb"]) > 0, ("reverb", path_dict.keys())
        y_srcs["reverb"] = np.stack(y_srcs["reverb"], axis=-1)

        # normalization:
        if self.normalization:
            y_mix, y_srcs = self._normalization(y_mix, y_srcs)

        return y_mix, y_srcs

    def _read_wav(self, path, start_samp=0, end_samp=None):
        y, sr = sf.read(path, start=start_samp, stop=end_samp, dtype=np.float32)
        assert sr == self.sr, f"samplerate of data {sr} does not match requested samplerate {self.sr}"
        assert y.ndim == 2 and y.shape[-1] > 1, f"audios must be multi-channel but the shape of {path} is {y.shape}"

        if self.channel_idx is not None:
            return y[..., self.channel_idx]
        else:
            return y

    def _random_start_and_end(self, path_dict, min_overlap=None):
        if min_overlap is None:
            min_overlap = self.chunk_size // 5

        assert "num_samples" in path_dict and "offset" in path_dict
        offset = max(path_dict["offset"])  # start of shorter utterance
        min_length = min(path_dict["num_samples"])  # length of shorter utterance
        max_length = max(path_dict["num_samples"])  # mixture length

        # when mixture is shorter than the chunk size we want,
        # we simply use the entire utterance
        if max_length < self.chunk_size:
            return 0, max_length

        # else, we randomly choose partially-overlapped segment
        # where at least "min_overlap"-length overlap exists
        left = offset
        right = left + min_length
        min_overlap = min(min_length // 2 - 1, min_overlap)

        assert left + min_overlap < right - min_overlap, (left, right, min_overlap)
        if np.random.random() > 0.5:
            start = np.random.randint(left + min_overlap, right - min_overlap)
            end = start + self.chunk_size
            if end > max_length:
                start -= end - max_length
                end = max_length
        else:
            end = np.random.randint(left + min_overlap, right - min_overlap)
            start = end - self.chunk_size
            if start < 0:
                start = 0
                end = self.chunk_size
        return start, end

    def _normalization(self, y_mix, y_srcs):
        # variance normalization
        mean = y_mix.mean(keepdims=True)
        std = y_mix.std(keepdims=True)

        y_mix = (y_mix - mean) / std
        if isinstance(y_srcs, dict):
            y_srcs["reverb"] = (y_srcs["reverb"] - mean[None]) / std[None]
        elif y_srcs is not None:
            y_srcs = (y_srcs - mean[None]) / std[None]

        return y_mix, y_srcs

    def _stft(self, samples):
        assert samples.ndim < 4
        # if there is channel dim
        if samples.ndim == 3:
            n_chan, n_src, n_samples = samples.shape
            samples = samples.reshape(-1, n_samples)
            reshaped = True
        else:
            reshaped = False
        X = do_stft(samples, **self.stft_conf)
        if reshaped:
            X = X.reshape(X.shape[:2] + (n_chan, n_src))
        return X

    def _rotate_and_stack_channels_stft(self, feat, srcs):
        """Rotate the channel order and stack.
        Suppose the input `feat` is (n_frame, n_freq, n_chan) and n_chan==2.
        This function stacks `feat` with the channel order of [0, 1] and [1, 0]
        and returns (n_frame, n_freq, n_chan, n_chan), which is used in ERAS.

        Returned `feat` is stacked along the batch dim in the `collate_seq_eras` funciton.
        """

        n_chan = len(self.channel_idx)  # number of microphone channels
        assert feat.shape[-1] == n_chan, feat.shape

        feat_stack = []
        for ref_channel in range(n_chan):
            feat_tmp = rotate_channels(feat, ref_channel, channel_dim=-1)
            feat_stack.append(feat_tmp)
        feat_stack = torch.stack(feat_stack, dim=-1)

        # first initialize the dict to make code cleaner
        srcs_stack = {}
        for src in srcs:
            if isinstance(srcs[src], dict) and src == "y_srcs":
                srcs_stack[src] = {}
                for key in srcs[src]:
                    srcs_stack[src][key] = []
            else:
                srcs_stack[src] = []

        for ref_channel in range(n_chan):
            for src in srcs:
                if isinstance(srcs[src], dict) and src == "y_srcs":
                    assert srcs[src]["reverb"].shape[-2] == n_chan, (
                        src,
                        key,
                        srcs[src]["reverb"].shape,
                    )
                    tmp = rotate_channels(srcs[src]["reverb"], ref_channel, channel_dim=-2)
                    srcs_stack[src]["reverb"].append(tmp)

                # process torch Tensor
                elif srcs[src] is not None:
                    assert srcs[src].shape[-1] == n_chan, (
                        src,
                        srcs[src].shape,
                    )
                    tmp = rotate_channels(srcs[src], ref_channel, channel_dim=-1)
                    srcs_stack[src].append(tmp)

        # stack along batch dim
        for src in srcs_stack:
            if isinstance(srcs[src], dict) and src == "y_srcs":
                srcs_stack[src]["reverb"] = torch.stack(srcs_stack[src]["reverb"], dim=-1)
            elif srcs[src] is not None:
                srcs_stack[src] = torch.stack(srcs_stack[src], dim=-1)

        return feat_stack, srcs_stack
