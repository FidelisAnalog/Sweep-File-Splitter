import numpy as np
from scipy.io.wavfile import read, write
from scipy.signal import find_peaks, sosfiltfilt, iirfilter
from scipy.ndimage import uniform_filter1d
import os
import logging
import matplotlib.pyplot as plt


# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)
fh = logging.StreamHandler()
fh_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
fh.setFormatter(fh_formatter)
logger.addHandler(fh)
logger.propagate = False


# User Parameters
INPUT_FILE = '20241022-T004_V15VMR_47k_305pF_10074A1.wav'
TEST_RECORD = 'TRS1007'  # Options: TRS1007, TRS1005, STR100


# Debug signal plots
def plot_signal(
    signal,
    Fs,
    normalized_signal=None,
    threshold=None,
    detected_end_time=None,
    detected_start_time=None,
    peaks=None,
    title="Signal Visualization",
):
    time = np.linspace(0, len(signal) / Fs, num=len(signal))  # Time axis

    plt.figure(figsize=(12, 6))
    plt.plot(time, signal, label="Signal")

    if normalized_signal is not None:
        plt.plot(time[: len(normalized_signal)], normalized_signal, label="Normalized Signal")

    if threshold is not None:
        plt.axhline(y=threshold, color="r", linestyle="--", label=f"Threshold = {threshold}")

    if detected_start_time is not None:
        plt.axvline(x=detected_start_time, color="b", linestyle="--", label="Detected Start")

    if detected_end_time is not None:
        plt.axvline(x=detected_end_time, color="g", linestyle="--", label="Detected End")

    if peaks is not None:
        plt.plot(time[peaks], signal[peaks], "x", label="Peaks")

    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)
    plt.show()


# Rotation Helper
def rotate_left(y_in, nd):
    return np.concatenate((y_in[nd:], y_in[:nd]))


# Read WAV File
def read_measurement(input_file):
    logging.info(f"Reading: {input_file}")
    Fs, y = read(input_file)
    logger.info(f"Sample Rate: {Fs}")
    if y.ndim < 2:
        raise ValueError("Input file must have at least two channels.")
    return y[:, 0], y[:, 1], Fs


# Write Output WAV File
def write_result(output_file, left, right, Fs, start_index, end_index):
    y = np.column_stack((left[start_index:end_index], right[start_index:end_index]))
    write(output_file, Fs, y)


# Bandpass Filter
def apply_bandpass(signal, low, high, Fs, order=17):
    sos = iirfilter(order, [low, high], rs=140, btype='band', analog=False, ftype='cheby2', fs=Fs, output='sos')
    return sosfiltfilt(sos, signal)


def find_end_of_burst(signal, Fs, threshold=0.01, lower_border=40, upper_border=50, consecutive_in_borders=10, shift_size=12, shiftings=3):
    # Detect peaks with constraints on minimum distance
    peaks, _ = find_peaks(signal, height=threshold, distance=lower_border)
    logger.debug(f"Peaks Found: {len(peaks)}")

    # Find valid sequences of peak spacing
    valid_diffs = (lower_border <= np.diff(peaks)) & (np.diff(peaks) <= upper_border)
    start_index = np.argmax(np.convolve(valid_diffs, np.ones(consecutive_in_borders, dtype=int), mode='valid') == consecutive_in_borders)
    if start_index == 0:
        raise ValueError("No valid burst found")

    start_sample = peaks[start_index]
    logger.debug(f"Start Index: {start_sample}")
    
    # Define burst region
    is_ = int(start_sample + (1 * Fs))  # Start 1s after the first peak
    ie = int(start_sample + (12 * Fs))  # End 12s after

    # Extract and smooth burst region
    cut_burst = signal[is_:ie]
    cut_burst = uniform_filter1d(cut_burst, size=shift_size * shiftings)

    # Normalize and find burst end
    cut_burst /= np.max(cut_burst)
    burst_end = np.argmax(cut_burst < threshold)

    end_sample = is_ + burst_end

    if logging.getLogger(__name__).isEnabledFor(logging.DEBUG):
        plot_signal(
            signal,
            Fs,
            peaks=peaks,
            threshold=threshold,
            detected_end_time=(end_sample / Fs),
            detected_start_time=(start_sample /Fs),
            title="Burst End Detection"
        )

    return end_sample


def find_end_of_sweep(sweep_start_sample, sweep_end_min, sweep_end_max, signal, Fs, threshold=0.05, shiftings=6):
    sample_offset_start = sweep_start_sample + int(Fs * sweep_end_min)
    sample_offset_end = sweep_start_sample + int(Fs * sweep_end_max)
    signal = signal[sample_offset_start:sample_offset_end]
    original_signal = signal

    logger.debug(f"Length of End Window: {len(signal)}")

    # Filter a bit by shifting and adding
    shiftings = 6
    signal_shifted = rotate_left(signal, 1)
    for i in range(shiftings):
        signal = signal + signal_shifted
        signal_shifted = rotate_left(signal_shifted, 1)

    # Find end    
    signal = np.array(signal < threshold, dtype=float)
    signal = np.diff(signal)
    end_sample = np.argmax(signal) + sample_offset_start

    logger.debug(f"End Sample (Global Index): {end_sample}")
    logger.debug(f"Sample Offset Start: {sample_offset_start}, Sample Offset End: {sample_offset_end}")
    logger.debug(f"End Sample (Relative Index): {end_sample - sample_offset_start}")

    if logging.getLogger(__name__).isEnabledFor(logging.DEBUG):
        plot_signal(
            original_signal,
            Fs,
            threshold=threshold,
            detected_end_time=(end_sample - sample_offset_start) / Fs,
            title="Sweep End Detection",
        )

    return end_sample


# Main Processing Function
def slice_audio(input_file, test_record):
    # Test record parameters
    record_params = {
        'TRS1007': {'sweep_offset': 74, 'sweep_end_min': 47, 'sweep_end_max': 55},
        'TRS1005': {'sweep_offset': 36, 'sweep_end_min': 30, 'sweep_end_max': 38},
        'STR100': {'sweep_offset': 74, 'sweep_end_min': 61, 'sweep_end_max': 69},
    }

    if test_record not in record_params:
        raise ValueError("Invalid test record.")

    params = record_params[test_record]

    # Read input file
    left, right, Fs = read_measurement(input_file)

    logger.info(f"Test Record: {test_record}")

    # Filter and maximize for end of pilot detection
    left_filtered = apply_bandpass(left, 500, 2000, Fs)
    left_normalized = np.abs(left_filtered) / np.max(np.abs(left_filtered))

    # Find end of first pilot tone
    start_left_sweep = find_end_of_burst(left_normalized, Fs)
    logger.info(f"Start of Left Sweep: {start_left_sweep}")

    # Find end of second pilot tone
    sample_offset = start_left_sweep + int(Fs * params['sweep_offset'])
    start_right_sweep = sample_offset + find_end_of_burst(left_normalized[sample_offset:], Fs)
    logger.info(f"Start of Right Sweep: {start_right_sweep}")

    # Filter and maximize for end of left sweep detection
    left_filtered = apply_bandpass(left, 10000, 40000, Fs)
    left_normalized = np.abs(left_filtered) / np.max(np.abs(left_filtered))

    # Find end of left sweep
    end_left_sweep = find_end_of_sweep(start_left_sweep, params['sweep_end_min'], params['sweep_end_max'], left_normalized, Fs)
    logger.info(f"End of Left Sweep: {end_left_sweep}")

    # Filter and maximize for end of right sweep detection
    right_filtered = apply_bandpass(right, 10000, 40000, Fs)
    right_normalized = np.abs(right_filtered) / np.max(np.abs(right_filtered))

    # Find end of right sweep
    end_right_sweep = find_end_of_sweep(start_right_sweep, params['sweep_end_min'], params['sweep_end_max'], right_normalized, Fs)
    logger.info(f"End of Right Sweep: {end_right_sweep}")

    logger.info(f"Left Sweep Duration: {(end_left_sweep-start_left_sweep)/Fs}")
    logger.info(f"Right Sweep Duration: {(end_right_sweep-start_right_sweep)/Fs}")

    if logging.getLogger(__name__).isEnabledFor(logging.DEBUG):
        plot_signal(left[start_left_sweep:end_left_sweep], Fs, title="Left Sweep Segment")
        plot_signal(right[start_right_sweep:end_right_sweep], Fs, title="Right Sweep Segment")

    
    # Write results
    output_file_left = os.path.splitext(input_file)[0] + '_L.wav'
    output_file_right = os.path.splitext(input_file)[0] + '_R.wav'

    logger.info(f"Writing {output_file_left}")
    write_result(output_file_left, left, right, Fs, start_left_sweep, end_left_sweep)

    logger.info(f"Writing {output_file_right}")
    write_result(output_file_right, right, left, Fs, start_right_sweep, end_right_sweep)
    
    logger.info("Processing Complete.")


# Execute Main Script
if __name__ == "__main__":
    slice_audio(INPUT_FILE, TEST_RECORD)
