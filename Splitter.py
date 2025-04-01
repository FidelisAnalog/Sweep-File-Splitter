import numpy as np
from scipy.io.wavfile import read, write
from scipy.signal import find_peaks, sosfiltfilt, iirfilter
from scipy.ndimage import uniform_filter1d
import os
import logging
import argparse
import matplotlib.pyplot as plt


__version__ = "1.0.12"


# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)
fh = logging.StreamHandler()
fh_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
fh.setFormatter(fh_formatter)
logger.addHandler(fh)
logger.propagate = False


# Get configuration from command-line arguments
def get_config():

    # Argument parser setup
    parser = argparse.ArgumentParser(description="Process parameters for Sweep File Splitter.")
    parser.add_argument("--file", type=str, help="Path to the input WAV file.", metavar ="")
    parser.add_argument("--test_record", type=str, help="Test record for extracting sweeps.")
    parser.add_argument("--no_save", action="store_false", help="Do not save extraced sweep files.")
    parser.add_argument('--version', action='version', version='Sweep File Splitter ' + __version__)
    parser.add_argument("--log_level", default='info', type=str, choices=['info', 'debug'], help="Change console logging level to DEBUG.")

    # Parse command-line arguments
    return vars(parser.parse_args())


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
        plt.plot(time[peaks], signal[peaks], "darkorange", label="Peaks")

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


# Filter
def apply_filter(signal, low, high, Fs, order=17, btype='band'):
    if btype == 'band':
        sos = iirfilter(order, [low, high], rs=140, btype='band', analog=False, ftype='cheby2', fs=Fs, output='sos')
        
    elif btype == 'high':
        sos = iirfilter(order, high, rs=140, btype='highpass', analog=False, ftype='cheby2', fs=Fs, output='sos')

    return sosfiltfilt(sos, signal)


def find_burst_bounds(signal, Fs, lower_border, upper_border, consecutive_in_borders=10, threshold=0.02, shift_size=12, shiftings=3):
    # Detect peaks with constraints on minimum distance
    peaks, _ = find_peaks(signal, height=threshold, distance=lower_border)#prominence=.5)
    logger.debug(f"Peaks Found: {len(peaks)}")

    # Find valid sequences of peak spacing
    valid_diffs = (lower_border <= np.diff(peaks)) & (np.diff(peaks) <= upper_border)
    start_index = np.argmax(np.convolve(valid_diffs, np.ones(consecutive_in_borders, dtype=int), mode='valid') == consecutive_in_borders)

    start_sample = peaks[start_index]
    logger.debug(f"Start Index: {start_sample}")
    
    # Define burst region
    is_ = int(start_sample + (1 * Fs))  # Start 1s after the first peak
    ie = int(start_sample + (14 * Fs))  # End 14s after

    # Extract and smooth burst region
    cut_burst = signal[is_:ie]
    cut_burst = uniform_filter1d(cut_burst, size=shift_size * shiftings)

    # Normalize and find burst end
    cut_burst /= np.max(cut_burst)
    burst_end = np.argmax(cut_burst < threshold)

    end_sample = is_ + burst_end

    logger.debug(f"End Index: {end_sample}")

    if logging.getLogger(__name__).isEnabledFor(logging.DEBUG):
        plot_signal(
            signal,
            Fs,
            peaks=peaks,
            threshold=threshold,
            detected_end_time=(end_sample / Fs),
            detected_start_time=(start_sample /Fs),
            title="Burst Detection"
        )

    return start_sample, end_sample


def find_end_of_sweep(sweep_start_sample, sweep_end_min, sweep_end_max, signal, Fs, threshold=0.05, shiftings=6):
    sample_offset_start = sweep_start_sample + int(Fs * sweep_end_min)
    sample_offset_end = sweep_start_sample + int(Fs * sweep_end_max)
    signal = signal[sample_offset_start:sample_offset_end]
    original_signal = signal

    logger.debug(f"Length of End Window: {len(signal)}")

    # Filter a bit by shifting and adding
    signal_shifted = rotate_left(signal, 1)
    for i in range(shiftings):
        signal = signal + signal_shifted
        signal_shifted = rotate_left(signal_shifted, 1)

    # Find end    
    signal = np.array(signal < threshold, dtype=float)
    signal = np.diff(signal)
    end_sample = np.argmax(signal) + sample_offset_start

    logger.debug(f"End Sample (Global Index): {end_sample}")
    logger.debug(f"Sample Offset Start: {sample_offset_start}, End: {sample_offset_end}")
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
def slice_audio(input_file, test_record, save_files):
    # Test record parameters
    record_params = {
        'TRS1007': {'sweep_offset': 74, 'sweep_end_min': 48, 'sweep_end_max': 52, 'sweep_start_detect': 0},
        'TRS1005': {'sweep_offset': 32, 'sweep_end_min': 26, 'sweep_end_max': 34, 'sweep_start_detect': 1},
        'STR100': {'sweep_offset': 74, 'sweep_end_min': 63, 'sweep_end_max': 67, 'sweep_start_detect': 0},
        'STR120': {'sweep_offset': 58, 'sweep_end_min': 45, 'sweep_end_max': 50, 'sweep_start_detect': 0},
        'STR130': {'sweep_offset': 82, 'sweep_end_min': 63, 'sweep_end_max': 67, 'sweep_start_detect': 0},
        'STR170': {'sweep_offset': 75, 'sweep_end_min': 63, 'sweep_end_max': 67, 'sweep_start_detect': 0},
        'QR2009': {'sweep_offset': 80, 'sweep_end_min': 48, 'sweep_end_max': 52, 'sweep_start_detect': 0},
        'QR2010': {'sweep_offset': 24, 'sweep_end_min': 15, 'sweep_end_max': 18, 'sweep_start_detect': 0},
        'XG7001': {'sweep_offset': 78, 'sweep_end_min': 48, 'sweep_end_max': 52, 'sweep_start_detect': 0},
        'XG7002': {'sweep_offset': 74, 'sweep_end_min': 26, 'sweep_end_max': 30, 'sweep_start_detect': 1},
        'XG7005': {'sweep_offset': 78, 'sweep_end_min': 48, 'sweep_end_max': 52, 'sweep_start_detect': 0},
        'DIN45543': {'sweep_offset': 78, 'sweep_end_min': 48, 'sweep_end_max': 52, 'sweep_start_detect': 0},
    'ИЗМ33С0327': {'sweep_offset': 58, 'sweep_end_min': 48, 'sweep_end_max': 52, 'sweep_start_detect': 0},

    }

    if test_record.upper() not in record_params:
        raise ValueError("Invalid test record.")

    params = record_params[test_record.upper()]

    # Read input file
    left, right, Fs = read_measurement(input_file)

    logger.info(f"Test Record: {test_record}")

    lower_border = int(Fs/2040) #=40@96k - have to scale with Fs
    upper_border = int(Fs/1960) #=50@96k

    # Filter and maximize for end of left pilot detection
    left_filtered = apply_filter(left, 500, 2000, Fs, btype='band')
    left_normalized = np.abs(left_filtered) / np.max(np.abs(left_filtered))

    # Find end of left pilot tone / start of sweep
    _, start_left_sweep = find_burst_bounds(left_normalized, Fs, lower_border, upper_border)

    if params['sweep_start_detect'] == 1:
        sample_offset = start_left_sweep + Fs
        start_left_sweep, _ = sample_offset + find_burst_bounds(left_normalized[sample_offset:], Fs, lower_border, upper_border)

    logger.info(f"Start of Left Sweep: {start_left_sweep}")

    # Filter and maximize for end of right pilot detection
    right_filtered = apply_filter(right, 500, 2000, Fs, btype='band')
    right_normalized = np.abs(right_filtered) / np.max(np.abs(right_filtered))

    # Find end of left pilot tone / start of sweep
    sample_offset = start_left_sweep + int(Fs * params['sweep_offset'])
    _, start_right_sweep = sample_offset + find_burst_bounds(right_normalized[sample_offset:], Fs, lower_border, upper_border)

    if params['sweep_start_detect'] == 1:
        sample_offset = start_right_sweep + Fs
        start_right_sweep, _ = sample_offset + find_burst_bounds(right_normalized[sample_offset:], Fs, lower_border, upper_border)

    logger.info(f"Start of Right Sweep: {start_right_sweep}")

    # Filter and maximize for end of left sweep detection
    left_filtered = apply_filter(left, None, 10000, Fs, btype='high')
    left_normalized = np.abs(left_filtered) / np.max(np.abs(left_filtered))

    # Find end of left sweep
    end_left_sweep = find_end_of_sweep(start_left_sweep, params['sweep_end_min'], params['sweep_end_max'], left_normalized, Fs)
    logger.info(f"End of Left Sweep: {end_left_sweep}")

    # Filter and maximize for end of right sweep detection
    right_filtered = apply_filter(right, None, 10000, Fs, btype='high')
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
    if save_files == 1:
        output_file_left = os.path.splitext(input_file)[0] + '_L.wav'
        output_file_right = os.path.splitext(input_file)[0] + '_R.wav'

        logger.info(f"Writing {output_file_left}")
        write_result(output_file_left, left, right, Fs, start_left_sweep, end_left_sweep)

        logger.info(f"Writing {output_file_right}")
        write_result(output_file_right, right, left, Fs, start_right_sweep, end_right_sweep)
    
    logger.info("Processing Complete.")


# Execute Main Script
if __name__ == "__main__":

    config = get_config()

    INPUT_FILE = config["file"]
    TEST_RECORD = config["test_record"]
    SAVE_FILES = config["no_save"]
    LOG_LEVEL = config["log_level"]

    if LOG_LEVEL.upper() == 'DEBUG':
        logger.setLevel(level=logging.DEBUG)
    if LOG_LEVEL.upper() == 'INFO':
        logger.setLevel(level=logging.INFO)

    slice_audio(INPUT_FILE, TEST_RECORD, SAVE_FILES)
