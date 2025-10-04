#!/usr/bin/env python3
"""
pd120_decode.py

Basic PD120 SSTV decoder (starter).  Not as robust as MMSSTV/QSSTV,
but decodes many PD120 recordings by:
 - detecting 1200 Hz sync pulses to find line starts
 - for each line, extracting three color scans (R,G,B)
 - mapping 1500-2300 Hz -> 0..255 brightness

Usage:
  python pd120_decode.py input.wav output.png

Dependencies:
  pip install numpy scipy pillow
"""
import sys
import numpy as np
from scipy.io import wavfile
from scipy.signal import stft, butter, filtfilt
from PIL import Image

# PD120 expected params (from references)
IMG_W = 640
IMG_H = 496  # PD family uses 640x496 (includes header lines)
LINE_SYNC_MS = 5.5225   # typical sync pulse length in ms (approx)
PORCH_MS = 1.5          # porch length (approx)
# PD120 total seconds ~126.103040 (Wikipedia) but we do line-sync based
# frequency->brightness mapping
F_MIN = 1500.0
F_MAX = 2300.0
SYNC_F = 1200.0

def hz_to_brightness(freq):
    """Map frequency (Hz) in [F_MIN,F_MAX] to 0..255"""
    v = (freq - F_MIN) / (F_MAX - F_MIN)
    v = np.clip(v, 0.0, 1.0)
    return np.uint8(v * 255)

def dominant_freq_from_stft(Zxx, freqs):
    """Return dominant frequency (Hz) per time frame using STFT magnitude."""
    mag = np.abs(Zxx)
    idx = np.argmax(mag, axis=0)
    return freqs[idx]

def bandpass(x, fs, low=1000, high=2600, order=4):
    b, a = butter(order, [low/(fs/2.0), high/(fs/2.0)], btype='band')
    return filtfilt(b, a, x)

def find_sync_times(audio, fs):
    """
    Very simple sync detector: short-time FFT; frames where dominant freq ~1200Hz
    and energy above threshold are considered sync candidates. Return frame indices.
    """
    nperseg = 1024
    f, t, Zxx = stft(audio, fs=fs, nperseg=nperseg, noverlap=nperseg- int(nperseg/2))
    dom = dominant_freq_from_stft(Zxx, f)
    mag = np.abs(Zxx).max(axis=0)
    # detect frames near SYNC_F
    is_sync = (np.abs(dom - SYNC_F) < 40) & (mag > np.percentile(mag, 60))
    # collapse nearby frames
    sync_times = []
    prev = -1
    for i, val in enumerate(is_sync):
        if val and (i - prev) > 2:
            sync_times.append(t[i])
            prev = i
    return np.array(sync_times)

def decode_lines(audio, fs, sync_times):
    """
    For each detected sync time, extract the line scan samples that follow and
    decode them into RGB values.
    PD120 lines contain (per spec approx):
      sync (~5.5ms at 1200Hz), porch (~1.5ms at 1500Hz), then R scan, G scan, B scan.
    We approximate scan durations so that one full line contains IMG_W pixels per color.
    """
    samples = len(audio)
    result_R = np.zeros((IMG_H, IMG_W), dtype=np.uint8)
    result_G = np.zeros((IMG_H, IMG_W), dtype=np.uint8)
    result_B = np.zeros((IMG_H, IMG_W), dtype=np.uint8)

    # estimate line period from sync spacing (median delta)
    if len(sync_times) < 10:
        print("Warning: few syncs detected, decoding may fail.")
    line_period = np.median(np.diff(sync_times)) if len(sync_times) > 1 else 0.734  # fallback
    # convert ms durations to seconds/ samples for our assumed layout
    sync_len = LINE_SYNC_MS / 1000.0
    porch_len = PORCH_MS / 1000.0
    # remaining time = total line period - sync - porch, divided by 3 scans
    if line_period <= (sync_len + porch_len + 0.0001):
        # fallback guess: total time per color scan based on PD120 timing
        # PD120: ~126s total / 496 lines = ~0.254 s per line (note: PD family timing is complex)
        # we'll attempt a reasonable scan_time
        scan_time = 0.073  # guess (seconds) — tune if wrong
    else:
        scan_time = (line_period - sync_len - porch_len) / 3.0

    # STFT parameters for high time resolution
    win = 1024
    hop = 512
    f, tframes, Zxx = stft(audio, fs=fs, nperseg=win, noverlap=win-hop)
    freqs = f
    domfreqs = dominant_freq_from_stft(Zxx, freqs)
    times = tframes  # center times of frames

    # helper to sample domfreqs in a time interval and map to brightness
    def decode_interval(t0, t1):
        # select frames between t0 and t1
        idx = np.where((times >= t0) & (times < t1))[0]
        if len(idx) == 0:
            return np.array([], dtype=np.uint8)
        freqs_segment = domfreqs[idx]
        # map to brightness
        brightness = hz_to_brightness(freqs_segment)
        return brightness

    # iterate syncs and decode lines; fill rows up to IMG_H
    row = 0
    for st in sync_times:
        if row >= IMG_H:
            break
        # compute times for the three scans
        t_r0 = st + sync_len + porch_len
        t_r1 = t_r0 + scan_time
        t_g0 = t_r1
        t_g1 = t_g0 + scan_time
        t_b0 = t_g1
        t_b1 = t_b0 + scan_time

        br = decode_interval(t_r0, t_r1)
        bg = decode_interval(t_g0, t_g1)
        bb = decode_interval(t_b0, t_b1)

        # We need exactly IMG_W pixels per color; resample (nearest) or interpolate
        def resample_to_width(arr):
            if arr.size == 0:
                return np.zeros(IMG_W, dtype=np.uint8)
            # interpolate indices
            src_x = np.linspace(0, 1, arr.size)
            dst_x = np.linspace(0, 1, IMG_W)
            return np.interp(dst_x, src_x, arr).astype(np.uint8)

        rowR = resample_to_width(br)
        rowG = resample_to_width(bg)
        rowB = resample_to_width(bb)

        result_R[row, :] = rowR
        result_G[row, :] = rowG
        result_B[row, :] = rowB

        row += 1

    # if fewer rows decoded than IMG_H, keep the decoded ones and crop or pad
    if row < IMG_H:
        print(f"Decoded {row} lines out of expected {IMG_H}.")
        result_R = result_R[:row, :]
        result_G = result_G[:row, :]
        result_B = result_B[:row, :]

    # merge channels
    rgb = np.dstack([result_R, result_G, result_B])
    return rgb

def main(wav_in, out_png):
    fs, data = wavfile.read(wav_in)
    if data.ndim > 1:
        data = data[:,0]  # take left channel if stereo
    # normalize to float
    audio = data.astype(np.float32)
    audio /= np.max(np.abs(audio)) + 1e-9

    # bandpass around expected SSTV audio
    audio_bp = bandpass(audio, fs, low=900, high=2600)

    print("Finding sync pulses ...")
    sync_times = find_sync_times(audio_bp, fs)
    print(f"Found {len(sync_times)} sync pulses (first few): {sync_times[:6]}")

    rgb = decode_lines(audio_bp, fs, sync_times)
    # convert to image and save
    img = Image.fromarray(rgb, mode='RGB')
    img.save(out_png)
    print("Saved:", out_png)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python pd120_decode.py input.wav output.png")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])