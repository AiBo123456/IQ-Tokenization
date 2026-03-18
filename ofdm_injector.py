import numpy as np


def generate_ofdm_signal(num_subcarriers=64, num_symbols=20, cp_length=16):
    """
    Generates a baseband complex OFDM signal using QPSK modulation.
    """
    # 1. Generate random QPSK symbols: (+/- 1 +/- j) / sqrt(2)
    symbols = (np.random.choice([-1, 1], size=(num_subcarriers, num_symbols)) +
               1j * np.random.choice([-1, 1], size=(num_subcarriers, num_symbols))) / np.sqrt(2)

    # 2. Null the DC subcarrier (Standard practice to avoid center-frequency leakage)
    symbols[0, :] = 0 + 0j

    # 3. Apply IFFT across subcarriers (axis 0) to get time-domain frames
    time_domain_symbols = np.fft.ifft(symbols, axis=0)

    # 4. Add Cyclic Prefix (CP) to mitigate inter-symbol interference
    cp = time_domain_symbols[-cp_length:, :]
    ofdm_frames = np.vstack((cp, time_domain_symbols))

    # 5. Serialize the parallel frames into a 1D complex time series
    ofdm_signal = ofdm_frames.flatten(order='F')

    return ofdm_signal


def inject_anomaly(base_signal, fs, ofdm_signal, freq_offset_hz=0, snr_db=10):
    """
    Shifts the OFDM signal in frequency, scales it, and injects it at a random time.
    """
    base_len = len(base_signal)
    ofdm_len = len(ofdm_signal)

    if ofdm_len > base_len:
        raise ValueError("Anomaly signal duration cannot exceed base signal duration.")

    # Pick a random starting index for the anomaly burst
    start_idx = np.random.randint(0, base_len - ofdm_len)

    # 1. Frequency shift the baseband OFDM signal to the desired offset
    t = np.arange(ofdm_len) / fs
    freq_shift = np.exp(1j * 2 * np.pi * freq_offset_hz * t)
    shifted_ofdm = ofdm_signal * freq_shift

    # 2. Scale for target SNR
    # Measure the background power specifically where the anomaly will land
    base_segment_power = np.mean(np.abs(base_signal[start_idx:start_idx + ofdm_len]) ** 2)
    ofdm_power = np.mean(np.abs(shifted_ofdm) ** 2)

    # Calculate scale factor (Treating anomaly as the 'Signal' and base as 'Noise')
    target_ofdm_power = base_segment_power * (10 ** (snr_db / 10))
    scaling_factor = np.sqrt(target_ofdm_power / ofdm_power)

    scaled_ofdm = shifted_ofdm * scaling_factor

    # 3. Inject into a copy of the base signal
    corrupted_signal = np.copy(base_signal)
    corrupted_signal[start_idx:start_idx + ofdm_len] += scaled_ofdm

    return corrupted_signal, start_idx


if __name__ == "__main__":
    # --- Demonstration ---

    fs = 1e6  # 1 MHz sampling rate
    duration = 0.1  # 100 ms total collection
    num_samples = int(fs * duration)

    # Generate a mock base signal (complex Additive White Gaussian Noise)
    np.random.seed(42)  # For reproducibility
    base_signal = (np.random.randn(num_samples) + 1j * np.random.randn(num_samples)) / np.sqrt(2)

    print(f"Base signal generated: {len(base_signal)} samples.")

    # Generate the synthetic OFDM anomaly
    # Bandwidth will be roughly (fs / num_subcarriers) * active_subcarriers
    anomaly_burst = generate_ofdm_signal(num_subcarriers=64, num_symbols=50, cp_length=16)

    print(f"OFDM anomaly generated: {len(anomaly_burst)} samples.")

    # Inject the anomaly at a +100 kHz offset with a 5 dB SNR relative to the background
    corrupted_signal, injection_index = inject_anomaly(
        base_signal=base_signal,
        fs=fs,
        ofdm_signal=anomaly_burst,
        freq_offset_hz=100000,
        snr_db=5
    )

    injection_time_ms = (injection_index / fs) * 1000
    print(f"Anomaly successfully injected at index {injection_index} ({injection_time_ms:.2f} ms).")