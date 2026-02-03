# ============================================================
# Global Configuration Parameters
# ============================================================

import numpy as np
import h5py
from scipy import signal
import time
import os
import argparse
from datetime import datetime
from multiprocessing import Pool, cpu_count
from functools import partial
from datetime import datetime
import matplotlib.pyplot as plt
from tqdm import tqdm

# File path configuration
FILE_PATH = "data/iq_data_for_export_20250709/hidden_level_iq_capture_1.mat"

# Sampling rate parameters
FS_ORIG = 163.84e6 / 160               # Original sampling rate: 163.84 MHz
DECIMATION_FACTOR = 1          # Decimation factor: ÷20
FS_DECIMATED = FS_ORIG / DECIMATION_FACTOR  # Post-decimation: 8.192 MHz

# Intermediate Frequency (IF) offset compensation
F_CENTER = 1550e6               # Data acquisition center frequency: 1550 MHz
F_GPS_L1 = 1575.42e6            # GPS L1 standard carrier frequency: 1575.42 MHz
F_IF_OFFSET = F_GPS_L1 - F_CENTER  # IF offset: +25.42 MHz (needs compensation)

# GPS C/A code parameters (from IS-GPS-200N standard)
CA_CHIP_RATE = 1.023e6          # C/A code chip rate: 1.023 MHz
CA_CODE_LENGTH = 1023           # C/A code length: 1023 chips

# Signal integration parameters
# INTEGRATION_MS = 20             # Non-coherent integration time: 20 milliseconds (improves SNR)
NUM_CHANNELS = 1                # Number of RF channels used: 8 channels

# Coarse search parameters (fast scan, larger step size)
COARSE_DOPPLER_MIN = -5000      # Doppler search minimum: -5000 Hz
COARSE_DOPPLER_MAX = 5000       # Doppler search maximum: +5000 Hz
COARSE_DOPPLER_STEP = 200       # Doppler step: 200 Hz (coarse search)

# Fine search parameters (refined scan, smaller step size)
FINE_DOPPLER_RANGE = 300        # Fine search Doppler range: ±300 Hz
FINE_DOPPLER_STEP = 25          # Doppler step: 25 Hz (fine search)

# PRN satellite search list
PRN_LIST = list(range(1, 33))   # Search PRN 1-32 (GPS standard satellites)

# Signal detection thresholds
DETECTION_THRESHOLD = 6.0       # Coarse search SNR detection threshold: 6σ
FINE_THRESHOLD = 6.0            # Fine search confirmation threshold: 6σ (adjustable: 4-10σ)

# Robust SNR computation parameters (exclude region around peak for noise estimation)
PEAK_EXCLUDE_DOPPLER = 2        # Exclude ±2 Doppler bins around peak
PEAK_EXCLUDE_CODE_PHASE = 16    # Exclude ±16 code phase samples around peak

# Parallel processing configuration
NUM_WORKERS = cpu_count()       # Default: use all CPU cores (can be specified via command line)

# ============================================================
# G2 Register Tap Selection Table (PRN 1-32)
# From IS-GPS-200N Table 3-Ia
# Used to generate the unique C/A code for each PRN
# ============================================================
G2_TAP_TABLE = {
    1:  (2, 6),   2:  (3, 7),   3:  (4, 8),   4:  (5, 9),
    5:  (1, 9),   6:  (2, 10),  7:  (1, 8),   8:  (2, 9),
    9:  (3, 10),  10: (2, 3),   11: (3, 4),   12: (5, 6),
    13: (6, 7),   14: (7, 8),   15: (8, 9),   16: (9, 10),
    17: (1, 4),   18: (2, 5),   19: (3, 6),   20: (4, 7),
    21: (5, 8),   22: (6, 9),   23: (1, 3),   24: (4, 6),
    25: (5, 7),   26: (6, 8),   27: (7, 9),   28: (8, 10),
    29: (1, 6),   30: (2, 7),   31: (3, 8),   32: (4, 9),
}

# ============================================================
# Signal Preprocessing Functions
# ============================================================

def apply_if_compensation(iq_data, f_if, fs):
    """
    Apply Intermediate Frequency (IF) offset compensation

    Principle: The center frequency during data acquisition differs from GPS L1 frequency,
               requiring frequency shifting for correction
    Method: Multiply signal by exp(-j*2π*f_if*t) to remove IF offset

    Parameters:
        iq_data: Input IQ data, shape=(8, N)
        f_if: IF offset frequency (Hz), positive value means downshift
        fs: Sampling rate (Hz)

    Returns:
        iq_compensated: IF-compensated IQ data, shape=(8, N)
    """
    num_channels, num_samples = iq_data.shape

    # Generate time vector: t = [0, 1/fs, 2/fs, ..., (N-1)/fs]
    t = np.arange(num_samples) / fs

    # Generate frequency shift phase: exp(-j*2π*f_if*t)
    # Negative sign indicates downshift (moving high-frequency signal to baseband)
    phase_shift = np.exp(-1j * 2 * np.pi * f_if * t)

    # Apply phase compensation to all channels
    iq_compensated = iq_data * phase_shift[np.newaxis, :]

    return iq_compensated


def decimate_complex(iq_data, factor, fs_orig):
    """
    Complex signal decimation (reduce sampling rate to decrease computation)

    Steps:
    1. Low-pass filtering (anti-aliasing)
    2. Downsampling (take every 'factor'-th sample)

    Parameters:
        iq_data: Input IQ data, shape=(8, N)
        factor: Decimation factor (e.g., 20 means sampling rate becomes 1/20 of original)
        fs_orig: Original sampling rate (Hz)

    Returns:
        iq_decimated: Decimated IQ data, shape=(8, N/factor)
    """
    num_channels = iq_data.shape[0]

    # Design low-pass filter
    # Cutoff frequency = (new sampling rate / 2) × 0.8 = (fs_orig / factor / 2) × 0.8
    # This preserves main frequency components while avoiding aliasing
    cutoff = (fs_orig / factor / 2) * 0.8

    # Use Chebyshev Type I filter (steep roll-off characteristic)
    sos = signal.cheby1(8, 0.05, cutoff / (fs_orig / 2), btype='low', output='sos')

    # Filter and downsample each channel separately
    iq_decimated = []
    for ch in range(num_channels):
        # Filter I and Q components separately (preserve phase relationship)
        i_filt = signal.sosfilt(sos, iq_data[ch, :].real)
        q_filt = signal.sosfilt(sos, iq_data[ch, :].imag)

        # Downsample: take every 'factor'-th sample
        i_dec = i_filt[::factor]
        q_dec = q_filt[::factor]

        # Recombine into complex numbers
        iq_decimated.append(i_dec + 1j * q_dec)

    return np.array(iq_decimated, dtype=np.complex64)


# ============================================================
# GPS C/A Code Generation Functions
# ============================================================

def generate_ca_code(prn):
    """
    Generate GPS C/A code (Gold code) for specified PRN

    Algorithm: Uses two 10-bit Linear Feedback Shift Registers (LFSR)
    - G1: Feedback polynomial 1 + X³ + X¹⁰
    - G2: Feedback polynomial 1 + X² + X³ + X⁶ + X⁸ + X⁹ + X¹⁰
    - C/A code = G1[10] ⊕ G2[tap1] ⊕ G2[tap2]

    Parameters:
        prn: PRN number (1-32)

    Returns:
        ca_code: C/A code sequence, length 1023, values ±1

    Reference: IS-GPS-200N Section 3.3.2.1
    """
    # Get G2 register tap positions for this PRN
    tap1, tap2 = G2_TAP_TABLE[prn]

    # Initialize both LFSRs (all-ones state)
    g1 = np.ones(10, dtype=int)
    g2 = np.ones(10, dtype=int)

    ca_code = []

    # Generate 1023 chips
    for _ in range(CA_CODE_LENGTH):
        # Output = last bit of G1 XOR two tap bits of G2
        # Note: Indices are 1-based, so subtract 1
        output = g1[9] ^ g2[tap1 - 1] ^ g2[tap2 - 1]

        # Convert to ±1 (0→+1, 1→-1)
        ca_code.append(1 if output == 0 else -1)

        # G1 feedback: X³ ⊕ X¹⁰ = g1[2] ⊕ g1[9]
        g1_feedback = g1[2] ^ g1[9]

        # G2 feedback: X² ⊕ X³ ⊕ X⁶ ⊕ X⁸ ⊕ X⁹ ⊕ X¹⁰
        g2_feedback = g2[1] ^ g2[2] ^ g2[5] ^ g2[7] ^ g2[8] ^ g2[9]

        # Shift (insert feedback bit at front, shift others right by one)
        g1 = np.concatenate(([g1_feedback], g1[:-1]))
        g2 = np.concatenate(([g2_feedback], g2[:-1]))

    return np.array(ca_code, dtype=np.float32)


def upsample_ca_code(ca_code, samples_per_chip):
    """
    (Retained for backward compatibility, but **no longer used** in acquisition pipeline)

    *Old implementation*: Repeats each chip an integer number of times, which introduces
    sampling rate mismatch and requires zero-padding.
    The acquisition pipeline now uses the precise time-axis mapping function:
    `generate_local_ca_for_acquisition`.
    """
    ca_upsampled = np.repeat(ca_code, samples_per_chip)
    return ca_upsampled


def generate_local_ca_for_acquisition(prn, fs, duration_ms=1):
    """
    Generate high-precision local C/A code for acquisition at given sampling rate fs.

    Key points:
    - Does not assume fs is an integer multiple of CA_CHIP_RATE
    - Directly computes chip index via time axis: chip index = floor(t * CA_CHIP_RATE) % 1023
    - Output length is strictly N = fs * duration_ms / 1000 (e.g., 8192 samples at 8.192 MHz for 1 ms)
    - No zero-padding, no phase drift, better suited for multi-millisecond non-coherent integration
    """
    # Original 1023-chip C/A code (±1)
    ca_basic = generate_ca_code(prn)  # shape=(1023,)

    # Target number of samples
    N = int(fs * duration_ms / 1000.0)

    # Precise time axis
    t = np.arange(N, dtype=np.float64) / fs  # seconds

    # Corresponding chip indices (continuous value), then take floor + mod 1023
    chip_indices = np.floor(t * CA_CHIP_RATE) % CA_CODE_LENGTH
    chip_indices = chip_indices.astype(np.int64)

    local_code = ca_basic[chip_indices]
    return local_code.astype(np.float32)


# ============================================================
# GPS Signal Acquisition Core Algorithms
# ============================================================

def run_acquisition_multi_ch_multi_ms(iq_data, ca_code, doppler_freqs,
                                       fs, integration_ms):
    """
    Multi-channel, multi-millisecond non-coherent integration GPS signal acquisition (FFT method)

    Principle:
    1. For each Doppler frequency:
       a. Generate Doppler compensation phase: exp(-j*2π*f_doppler*t)
       b. Multiply received signal by compensation phase (remove Doppler shift)
       c. Use FFT for fast correlation (search all code phases)
    2. Non-coherent integration: Accumulate correlation power from multiple 1 ms segments (improve SNR)
    3. Multi-channel integration: Accumulate results from 8 RF channels (further improve SNR)

    Parameters:
        iq_data: IQ data, shape=(8_channels, N_samples)
        ca_code: Local C/A code (length = samples per 1 ms)
        doppler_freqs: Array of Doppler frequencies (Hz)
        fs: Sampling rate (Hz)
        integration_ms: Integration time (ms)

    Returns:
        metric_2d: 2D search space matrix, shape=(len(doppler_freqs), len(ca_code))
                   Each element represents correlation power at that (Doppler, code phase) combination
    """
    num_channels, total_samples = iq_data.shape
    samples_per_ms = int(fs / 1000)  # Number of samples per millisecond
    num_ms = integration_ms          # Number of milliseconds for integration

    # Pre-compute FFT of C/A code (for fast correlation)
    # Note: correlation = IFFT(FFT(signal) × conj(FFT(ca_code)))
    ca_fft = np.fft.fft(ca_code)

    # Initialize 2D search space (Doppler × code phase)
    metric_2d = np.zeros((len(doppler_freqs), len(ca_code)))

    # Search for each Doppler frequency
    for d_idx, f_doppler in enumerate(doppler_freqs):
        # Accumulator: non-coherent integration across time segments and channels
        correlation_power = np.zeros(len(ca_code))

        # Process each time segment (1 ms)
        for ms_idx in range(num_ms):
            start_idx = ms_idx * samples_per_ms
            end_idx = start_idx + len(ca_code)

            # Process each RF channel
            for ch in range(num_channels):
                # Extract current time segment signal
                signal_seg = iq_data[ch, start_idx:end_idx]

                # Generate Doppler compensation phase
                t = np.arange(len(signal_seg)) / fs + ms_idx / 1000.0
                doppler_phase = np.exp(-1j * 2 * np.pi * f_doppler * t)

                # Apply Doppler compensation
                signal_compensated = signal_seg * doppler_phase

                # FFT fast correlation
                signal_fft = np.fft.fft(signal_compensated)
                correlation = np.fft.ifft(signal_fft * np.conj(ca_fft))

                # Non-coherent integration: accumulate correlation power (magnitude squared)
                correlation_power += np.abs(correlation) ** 2

        # Store correlation power curve for this Doppler frequency
        metric_2d[d_idx, :] = correlation_power

    return metric_2d


def compute_robust_snr(metric_2d, peak_d_idx, peak_c_idx,
                       exclude_doppler=2, exclude_code=16):
    """
    Compute robust SNR (Signal-to-Noise Ratio)

    Method:
    1. Peak power = metric_2d[peak_d_idx, peak_c_idx]
    2. Noise power = Average power after excluding region around peak
    3. SNR = (peak power - noise power) / std(noise power)  Unit: σ (standard deviation)
    """
    num_doppler, num_code = metric_2d.shape

    # Create mask: mark which regions should be excluded (True = exclude)
    mask = np.zeros_like(metric_2d, dtype=bool)

    # Determine exclusion region boundaries (handle array boundaries)
    d_min = max(0, peak_d_idx - exclude_doppler)
    d_max = min(num_doppler, peak_d_idx + exclude_doppler + 1)
    c_min = max(0, peak_c_idx - exclude_code)
    c_max = min(num_code, peak_c_idx + exclude_code + 1)

    # Mark region around peak as True (to be excluded)
    mask[d_min:d_max, c_min:c_max] = True

    # Extract noise samples (excluding peak region)
    noise_samples = metric_2d[~mask]

    # Compute noise statistics
    noise_mean = np.mean(noise_samples)
    noise_std = np.std(noise_samples)

    # Compute SNR (unit: standard deviation σ)
    peak_power = metric_2d[peak_d_idx, peak_c_idx]
    snr = (peak_power - noise_mean) / noise_std if noise_std > 0 else 0.0

    return snr


def run_fine_acquisition(iq_data, ca_code, coarse_doppler, fs, integration_ms,
                          fine_range=300, fine_step=25):
    """
    Fine search: Perform more refined Doppler search around coarse search result
    """
    # Define fine search Doppler frequency range
    fine_doppler_min = coarse_doppler - fine_range
    fine_doppler_max = coarse_doppler + fine_range
    fine_doppler_freqs = np.arange(fine_doppler_min, fine_doppler_max + fine_step, fine_step)

    # Run fine search
    fine_metric_2d = run_acquisition_multi_ch_multi_ms(
        iq_data, ca_code, fine_doppler_freqs, fs, integration_ms
    )

    return fine_metric_2d, fine_doppler_freqs


def parabolic_interpolation_2d(metric_2d, peak_d_idx, peak_c_idx):
    """
    2D parabolic interpolation: Obtain sub-sample peak position
    """
    num_doppler, num_code = metric_2d.shape
    delta_d = 0.0
    delta_c = 0.0

    # Doppler direction interpolation
    if 0 < peak_d_idx < num_doppler - 1:
        y_prev = metric_2d[peak_d_idx - 1, peak_c_idx]
        y_peak = metric_2d[peak_d_idx, peak_c_idx]
        y_next = metric_2d[peak_d_idx + 1, peak_c_idx]

        denom = 2 * (y_prev - 2 * y_peak + y_next)
        if denom != 0:
            delta_d = (y_prev - y_next) / denom

    # Code phase direction interpolation
    if 0 < peak_c_idx < num_code - 1:
        y_prev = metric_2d[peak_d_idx, peak_c_idx - 1]
        y_peak = metric_2d[peak_d_idx, peak_c_idx]
        y_next = metric_2d[peak_d_idx, peak_c_idx + 1]

        denom = 2 * (y_prev - 2 * y_peak + y_next)
        if denom != 0:
            delta_c = (y_prev - y_next) / denom

    return delta_d, delta_c


def evaluate_signal_quality(metric_2d, peak_d_idx, peak_c_idx):
    """
    Evaluate signal quality: Peak sharpness & second peak ratio
    """
    # Peak power
    peak_value = metric_2d[peak_d_idx, peak_c_idx]

    # Compute peak sharpness = peak value / mean value of that row
    row_mean = np.mean(metric_2d[peak_d_idx, :])
    sharpness = peak_value / row_mean if row_mean > 0 else 0.0

    # Compute second peak ratio
    metric_copy = metric_2d.copy()

    # Exclude ±16 code phase samples around peak
    c_min = max(0, peak_c_idx - 16)
    c_max = min(metric_2d.shape[1], peak_c_idx + 16 + 1)
    metric_copy[peak_d_idx, c_min:c_max] = 0

    # Find second largest peak
    second_peak_value = np.max(metric_copy)
    second_peak_ratio = peak_value / second_peak_value if second_peak_value > 0 else 0.0

    return sharpness, second_peak_ratio


# ============================================================
# Physical Quantity Computation Functions
# ============================================================

def code_phase_to_meters(code_phase_samples, fs):
    """
    Convert code phase (samples) to "fractional pseudorange" (meters)
    This is only c × τ (mod 1 ms), not a complete pseudorange solution.
    """
    C_LIGHT = 299792458.0  # Speed of light m/s

    # Time delay = number of samples / sampling rate
    delay_sec = code_phase_samples / fs

    # Distance = speed of light × time
    pseudorange_m = C_LIGHT * delay_sec

    return pseudorange_m


def doppler_to_velocity(f_doppler):
    """
    Convert Doppler frequency shift to radial velocity
    """
    C_LIGHT = 299792458.0  # Speed of light m/s

    # Velocity = Doppler shift × speed of light / carrier frequency
    # Note negative sign: positive Doppler means frequency increase, i.e., satellite moving away
    velocity_mps = -f_doppler * C_LIGHT / F_GPS_L1

    return velocity_mps



# ============================================================
# Parallel Processing Helper Functions
# ============================================================

def process_single_prn_coarse(args):
    """
    Single PRN coarse search processing function (for parallelization)
    """
    (prn, iq_decimated, ca_code, coarse_doppler_freqs,
     fs_decimated, integration_ms, detection_threshold,
     peak_exclude_doppler, peak_exclude_code_phase) = args

    metric_2d = run_acquisition_multi_ch_multi_ms(
        iq_decimated, ca_code, coarse_doppler_freqs,
        fs_decimated, integration_ms
    )

    peak_d_idx, peak_c_idx = np.unravel_index(np.argmax(metric_2d), metric_2d.shape)

    snr = compute_robust_snr(metric_2d, peak_d_idx, peak_c_idx,
                             peak_exclude_doppler, peak_exclude_code_phase)

    if snr >= detection_threshold:
        return {
            'prn': prn,
            'doppler_idx': peak_d_idx,
            'code_idx': peak_c_idx,
            'doppler': coarse_doppler_freqs[peak_d_idx],
            'snr': snr,
            'metric_2d': metric_2d
        }
    return None


def process_single_prn_fine(args):
    """
    Single PRN fine search processing function (for parallelization)
    """
    (res, iq_decimated, ca_code, fs_decimated, integration_ms,
     fine_doppler_range, fine_doppler_step, fine_threshold,
     samples_per_ms) = args

    prn = res['prn']

    fine_metric_2d, fine_doppler_freqs = run_fine_acquisition(
        iq_decimated, ca_code, res['doppler'],
        fs_decimated, integration_ms,
        fine_doppler_range, fine_doppler_step
    )

    fine_peak_d_idx, fine_peak_c_idx = np.unravel_index(
        np.argmax(fine_metric_2d), fine_metric_2d.shape
    )

    delta_d, delta_c = parabolic_interpolation_2d(
        fine_metric_2d, fine_peak_d_idx, fine_peak_c_idx
    )

    fine_doppler = fine_doppler_freqs[fine_peak_d_idx] + delta_d * fine_doppler_step
    fine_code_phase = fine_peak_c_idx + delta_c

    fine_snr = compute_robust_snr(fine_metric_2d, fine_peak_d_idx, fine_peak_c_idx)
    sharpness, second_peak_ratio = evaluate_signal_quality(
        fine_metric_2d, fine_peak_d_idx, fine_peak_c_idx
    )

    cn0_estimate = 10 * np.log10(fine_snr ** 2 / (integration_ms / 1000.0))
    pseudorange_m = code_phase_to_meters(fine_code_phase, fs_decimated)
    velocity_mps = doppler_to_velocity(fine_doppler)
    chips = fine_code_phase / samples_per_ms * 1023
    confirmed = fine_snr >= fine_threshold

    return {
        'prn': prn,
        'coarse_doppler': res['doppler'],
        'coarse_code_phase': res['code_idx'],
        'coarse_snr': res['snr'],
        'fine_doppler': fine_doppler,
        'fine_code_phase': fine_code_phase,
        'fine_snr': fine_snr,
        'chips': chips,
        'pseudorange_m': pseudorange_m,
        'velocity_mps': velocity_mps,
        'cn0_estimate': cn0_estimate,
        'sharpness': sharpness,
        'second_peak_ratio': second_peak_ratio,
        'confirmed': confirmed,
        'fine_metric_2d': fine_metric_2d,
        'fine_doppler_freqs': fine_doppler_freqs
    }


def main():
    file_path = "IQ_labeled_data/July_downsampling/"
    dataset_names = ["train_revin_x.npy", "val_revin_x.npy", "test_revin_x.npy"]
    # dataset_names = ["train_notrevin_x.npy"]
    workers = 4

    for dataset_name in dataset_names:
        data = np.load(os.path.join(file_path, dataset_name))
        iq_complex = (data[:, 0] + 1j * data[:, 1]).astype(np.complex64)
        print(f"{dataset_name} shape: {data.shape}, IQ complex shape: {iq_complex.shape}")
        INTEGRATION_MS = int(iq_complex.shape[-1] * 1000 // FS_DECIMATED)
        # print("INTEGRATION_MS:", INTEGRATION_MS)
        stats = []
        SNRs = []

        for iq_data in tqdm(iq_complex):
            iq_data = iq_data[np.newaxis, ...]  

            # Calculate post-decimation parameters
            samples_per_ms = int(FS_DECIMATED / 1000)

            # Define coarse search Doppler frequencies
            coarse_doppler_freqs = np.arange(COARSE_DOPPLER_MIN, COARSE_DOPPLER_MAX + COARSE_DOPPLER_STEP,
                                            COARSE_DOPPLER_STEP)

            # Generate local C/A codes for all PRNs (**precise time-axis version**)
            ca_codes = {}
            for prn in PRN_LIST:
                ca_local = generate_local_ca_for_acquisition(
                    prn=prn,
                    fs=FS_DECIMATED,
                    duration_ms=1
                )
                # sanity check: length should equal samples_per_ms
                if len(ca_local) != samples_per_ms:
                    raise RuntimeError(
                        f"Local CA code length mismatch for PRN {prn}: "
                        f"{len(ca_local)} vs expected {samples_per_ms}"
                    )
                ca_codes[prn] = ca_local

            # Coarse search - **Multi-process parallel**
            coarse_start_time = time.time()

            # Prepare argument list
            args_list = [
                (prn, iq_data, ca_codes[prn], coarse_doppler_freqs,
                FS_DECIMATED, INTEGRATION_MS, DETECTION_THRESHOLD,
                PEAK_EXCLUDE_DOPPLER, PEAK_EXCLUDE_CODE_PHASE)
                for prn in PRN_LIST
            ]

            # Parallel processing
            # print(f"  Using {workers} worker processes...")
            with Pool(processes=workers) as pool:
                results = pool.map(process_single_prn_coarse, args_list)

            # Filter out None results
            coarse_results = [r for r in results if r is not None]

            # coarse_time = time.time() - coarse_start_time
            # print(f"  Coarse search completed in {coarse_time:.2f}s ({workers} workers)")
            # print(f"  Found {len(coarse_results)} candidates")
            # print()


            # print(f"  Fine Doppler range: +/-{FINE_DOPPLER_RANGE} Hz, step {FINE_DOPPLER_STEP} Hz")
            # print(f"  Confirmation threshold: {FINE_THRESHOLD}σ")

            # fine_start_time = time.time()

            # Prepare fine search argument list
            fine_args_list = [
                (res, iq_data, ca_codes[res['prn']], FS_DECIMATED, INTEGRATION_MS,
                FINE_DOPPLER_RANGE, FINE_DOPPLER_STEP, FINE_THRESHOLD, samples_per_ms)
                for res in coarse_results
            ]

            # Parallel fine search processing
            with Pool(processes=workers) as pool:
                final_results = pool.map(process_single_prn_fine, fine_args_list)

            # fine_time = time.time() - fine_start_time
            # print(f"  Fine search completed in {fine_time:.2f}s ({workers} workers)")
            # print()

            # ========================================
            # Step 4: Results output
            # ========================================
            # print("[Step 4/4] Results")
            # print("=" * 90)

            confirmed_sats = [r for r in final_results if r['confirmed']]
            # print(f"Detected Satellites: {len(final_results)}")
            # print(f"Confirmed Satellites: {len(confirmed_sats)}")
            # print()

            # if confirmed_sats:
            #     print("Confirmed Satellites:")
            #     print("-" * 90)
            #     print(f"{'PRN':>4} {'Doppler (Hz)':>12} {'SNR (σ)':>10} {'C/N0 (dB-Hz)':>14} "
            #         f"{'Velocity (m/s)':>15} {'Status':>10}")
            #     print("-" * 90)

            #     for r in sorted(confirmed_sats, key=lambda x: x['fine_snr'], reverse=True):
            #         print(f"{r['prn']:>4} {r['fine_doppler']:>+12.1f} {r['fine_snr']:>10.2f} "
            #             f"{r['cn0_estimate']:>14.1f} {r['velocity_mps']:>+15.1f} "
            #             f"{'Confirmed':>10}")
            # else:
            #     print("Warning: No satellites confirmed (try lowering FINE_THRESHOLD)")

            # print()
            # print(f"Total processing time: {coarse_time + fine_time:.2f}s")
            # print("=" * 90)
            # print()

            stats_result = sorted(confirmed_sats, key=lambda x: x['fine_snr'], reverse=True)
            stats.append(stats_result[0]['prn'])
            SNRs.append(stats_result[0]['fine_snr'])
            # print(f"PRN: {stats_result[0]['prn']}, SNR: {stats_result[0]['fine_snr']:.2f}")

        np.savez(file_path+dataset_name.split('.')[0]+"labels.npz", stats=stats, SNRs=SNRs)

        # data = np.array(stats)

        # mean = data.mean()
        # std = data.std()

        # plt.figure()
        # plt.hist(data, bins=30, density=True, alpha=0.7)

        # plt.axvline(mean, linestyle='--', label='Mean')
        # plt.axvline(mean - std, linestyle=':', label='Mean ± Std')
        # plt.axvline(mean + std, linestyle=':')

        # plt.xlabel("Value")
        # plt.ylabel("Density")
        # plt.title("Distribution of Satellite PRNs")

        # data = np.array(SNRs)

        # mean = data.mean()
        # std = data.std()

        # plt.figure()
        # plt.hist(data, bins=30, density=True, alpha=0.7)

        # plt.axvline(mean, linestyle='--', label='Mean')
        # plt.axvline(mean - std, linestyle=':', label='Mean ± Std')
        # plt.axvline(mean + std, linestyle=':')

        # plt.xlabel("Value")
        # plt.ylabel("Density")
        # plt.title("Distribution of Satellite SNRs")
        # plt.show()


if __name__ == "__main__":
    main()