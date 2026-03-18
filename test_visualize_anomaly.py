import numpy as np
import matplotlib.pyplot as plt
from ofdm_injector import generate_ofdm_signal, inject_anomaly


def run_tutorial_test():
    print("Setting up test parameters...")
    fs = 1e6  # 1 MHz sampling rate
    duration = 0.1  # 100 ms
    num_samples = int(fs * duration)

    # Generate baseline complex noise
    np.random.seed(42)
    base_signal = (np.random.randn(num_samples) + 1j * np.random.randn(num_samples)) / np.sqrt(2)

    # Generate the OFDM burst
    anomaly = generate_ofdm_signal(num_subcarriers=64, num_symbols=50, cp_length=16)

    print("Injecting anomaly...")
    # Inject at +200 kHz with a +10 dB SNR so it stands out clearly in the plot
    freq_offset = 200000
    snr_db = 10
    corrupted_signal, start_idx = inject_anomaly(
        base_signal, fs, anomaly, freq_offset_hz=freq_offset, snr_db=snr_db
    )

    # --- Unit Tests ---
    print("Running assertions...")
    assert len(corrupted_signal) == len(base_signal), "Length mismatch after injection!"
    assert not np.array_equal(base_signal, corrupted_signal), "Signal was not modified!"
    print("Assertions passed. Generating spectrograms...\n")

    # --- Visualization ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, sharey=True)

    # Spectrogram parameters
    nfft = 256
    noverlap = 128

    # Plot Baseline
    ax1.specgram(base_signal, NFFT=nfft, Fs=fs, Fc=0, noverlap=noverlap, cmap='viridis')
    ax1.set_title('Baseline Signal (Complex AWGN)')
    ax1.set_ylabel('Frequency (Hz)')

    # Plot Corrupted Signal
    ax2.specgram(corrupted_signal, NFFT=nfft, Fs=fs, Fc=0, noverlap=noverlap, cmap='viridis')
    ax2.set_title(f'Corrupted Signal (OFDM Anomaly at {freq_offset / 1000} kHz)')
    ax2.set_xlabel('Time (Seconds)')
    ax2.set_ylabel('Frequency (Hz)')

    # Highlight the anomaly location
    start_time = start_idx / fs
    end_time = (start_idx + len(anomaly)) / fs
    ax2.axvline(start_time, color='red', linestyle='--', alpha=0.7)
    ax2.axvline(end_time, color='red', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_tutorial_test()