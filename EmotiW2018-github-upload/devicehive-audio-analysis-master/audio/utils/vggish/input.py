# Copyright (C) 2017 DataArt
#
# Based on
#
# Copyright 2017 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import resampy

from audio import params
from . import mel_features


def waveform_to_examples(data, sample_rate):
    """Converts audio waveform into an array of examples for VGGish.

    Args:
      data: np.array of either one dimension (mono) or two dimensions
        (multi-channel, with the outer dimension representing channels).
        Each sample is generally expected to lie in the range [-1.0, +1.0],
        although this is not required.
      sample_rate: Sample rate of data.

    Returns:
      3-D np.array of shape [num_examples, num_frames, num_bands] which represents
      a sequence of examples, each of which contains a patch of log mel
      spectrogram, covering num_frames frames of audio and num_bands mel frequency
      bands, where the frame length is params.STFT_HOP_LENGTH_SECONDS.
    """
    # Convert to mono.
    if len(data.shape) > 1:
        data = np.mean(data, axis=1)
    # Resample to the rate assumed by VGGish.
    if sample_rate != params.SAMPLE_RATE:
        data = resampy.resample(data, sample_rate, params.SAMPLE_RATE)

    # Compute log mel spectrogram features.
    log_mel = mel_features.log_mel_spectrogram(
        data,
        audio_sample_rate=params.SAMPLE_RATE, # 16000
        log_offset=params.LOG_OFFSET, # 0.01 Offset used for stabilized log of input mel-spectrogram.
        window_length_secs=params.STFT_WINDOW_LENGTH_SECONDS, # 25ms
        hop_length_secs=params.STFT_HOP_LENGTH_SECONDS, # 10ms
        num_mel_bins=params.NUM_MEL_BINS, # 64 # Frequency bands in input mel-spectrogram patch.
        lower_edge_hertz=params.MEL_MIN_HZ, # 125Hz
        upper_edge_hertz=params.MEL_MAX_HZ) # 7500Hz

    # Frame features into examples.
    features_sample_rate = 1.0 / params.STFT_HOP_LENGTH_SECONDS # 100 frame/s
    example_window_length = int(round(
        params.EXAMPLE_WINDOW_SECONDS * features_sample_rate)) # 0.96  # Each example contains 96 10ms frames
    example_hop_length = int(round(
        params.EXAMPLE_HOP_SECONDS * features_sample_rate)) # 0.96 # with zero overlap.
    log_mel_examples = mel_features.frame( # [3, 96, 64] change log-mel to batch whose length is 1s
        log_mel,
        window_length=example_window_length, # 96
        hop_length=example_hop_length) # 96
    return log_mel_examples
