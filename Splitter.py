import numpy as np
from scipy.io.wavfile import read, write
from scipy.signal import sosfiltfilt, hilbert, butter
from scipy.ndimage import uniform_filter1d
import os
import logging
import argparse
import matplotlib.pyplot as plt


__version__ = "1.1.0"


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


def find_burst_bounds(signal, Fs, tone_freq=1000, min_duration=1.0, threshold=0.3, search_duration=30.0):
    """
    Find pilot tone burst using Hilbert envelope method.
    More robust and sample-rate independent than peak-spacing method.
    
    Parameters:
    - signal: input audio signal
    - Fs: sample rate
    - tone_freq: expected pilot tone frequency (default 1000 Hz)
    - min_duration: minimum duration in seconds for valid burst (default 1.0s)
    - threshold: normalized envelope threshold (default 0.3 = 30% of peak)
    - search_duration: duration to search in seconds (default 20s)
    
    Returns:
    - start_sample: sample index where burst starts
    - end_sample: sample index where burst ends
    """
    # Only process first search_duration seconds to keep processing fast
    search_samples = int(search_duration * Fs)
    if len(signal) > search_samples:
        logger.debug(f"Limiting search to first {search_duration}s ({search_samples} samples)")
        signal_search = signal[:search_samples]
    else:
        signal_search = signal
    
    # Bandpass filter around tone frequency (±50 Hz tolerance)
    sos = butter(4, [tone_freq - 50, tone_freq + 50], btype='band', fs=Fs, output='sos')
    filtered = sosfiltfilt(sos, signal_search)
    
    # Hilbert transform to get analytic signal and envelope
    analytic_signal = hilbert(filtered)
    envelope = np.abs(analytic_signal)
    
    # Fast envelope smoothing using uniform_filter1d
    window_size = int(0.1 * Fs)
    envelope_smooth = uniform_filter1d(envelope, size=window_size, mode='nearest')
    
    # Normalize envelope
    envelope_norm = envelope_smooth / np.max(envelope_smooth)
    
    # Threshold detection
    above_threshold = envelope_norm > threshold
    
    # Find transitions (rising and falling edges)
    transitions = np.diff(above_threshold.astype(int))
    starts = np.where(transitions == 1)[0]
    ends = np.where(transitions == -1)[0]
    
    # Find first sustained region above threshold
    min_samples = int(min_duration * Fs)
    
    for start, end in zip(starts, ends):
        duration = end - start
        if duration >= min_samples:
            logger.debug(f"Found burst: start={start}, end={end}, duration={duration/Fs:.2f}s")
            
            if logging.getLogger(__name__).isEnabledFor(logging.DEBUG):
                plot_signal(
                    signal_search,
                    Fs,
                    normalized_signal=envelope_norm,
                    threshold=threshold,
                    detected_start_time=start / Fs,
                    detected_end_time=end / Fs,
                    title="Hilbert Envelope Burst Detection"
                )
            
            return start, end
    
    raise ValueError(f"No sustained pilot tone found in first {search_duration}s (min duration: {min_duration}s, threshold: {threshold})")


def find_sweep_start(signal, Fs, search_duration=20.0, threshold=0.2):
    """
    Find the start of a frequency sweep (rising energy), not a sustained tone.
    Used for test records where sweep starts several seconds after pilot tone ends.
    
    Parameters:
    - signal: raw audio signal
    - Fs: sample rate  
    - search_duration: how long to search (seconds)
    - threshold: energy rise threshold (default 0.2 = 20% of max)
    
    Returns:
    - start_sample: where sweep energy begins to rise
    """
    search_samples = int(search_duration * Fs)
    if len(signal) > search_samples:
        signal_search = signal[:search_samples]
    else:
        signal_search = signal
    
    # Bandpass around 1kHz (sweep typically starts at 1kHz)
    sos = butter(4, [900, 1100], btype='band', fs=Fs, output='sos')
    filtered = sosfiltfilt(sos, signal_search)
    
    # Get envelope
    envelope = np.abs(hilbert(filtered))
    
    # Smooth with larger window to see overall energy trend
    window_size = int(0.2 * Fs)  # 200ms window
    envelope_smooth = uniform_filter1d(envelope, size=window_size, mode='nearest')
    
    # Normalize
    envelope_norm = envelope_smooth / np.max(envelope_smooth)
    
    # Find where energy rises above threshold
    above_threshold = envelope_norm > threshold
    
    # Find first rising edge
    transitions = np.diff(above_threshold.astype(int))
    rises = np.where(transitions == 1)[0]
    
    if len(rises) > 0:
        start_sample = rises[0]
        logger.debug(f"Found sweep start at sample {start_sample} ({start_sample/Fs:.2f}s)")
        
        if logging.getLogger(__name__).isEnabledFor(logging.DEBUG):
            plot_signal(
                signal_search,
                Fs,
                normalized_signal=envelope_norm,
                threshold=threshold,
                detected_start_time=start_sample / Fs,
                title="Sweep Start Detection (Energy Rise at 1kHz)"
            )
        
        return start_sample
    else:
        raise ValueError(f"No sweep start found in first {search_duration}s")



def find_end_of_sweep(sweep_start_sample, sweep_end_min, sweep_end_max, signal, Fs, threshold=0.05):
    """
    Find end of frequency sweep using Hilbert envelope - optimized and automatic.
    Works for sweeps ending anywhere from 10kHz to 75kHz without configuration.
    
    Parameters:
    - sweep_start_sample: sample where sweep starts
    - sweep_end_min: minimum expected sweep duration (seconds)
    - sweep_end_max: maximum expected sweep duration (seconds)
    - signal: raw audio signal (not pre-filtered)
    - Fs: sample rate
    - threshold: relative amplitude threshold for end detection (default 0.05 = 5%)
    
    Returns:
    - end_sample: sample index where sweep ends
    """
    # Define search window
    sample_offset_start = sweep_start_sample + int(Fs * sweep_end_min)
    sample_offset_end = sweep_start_sample + int(Fs * sweep_end_max)
    signal_window = signal[sample_offset_start:sample_offset_end]
    
    logger.debug(f"End search window: {len(signal_window)} samples ({len(signal_window)/Fs:.2f}s)")
    
    # Use a moderate highpass (5kHz) to catch energy from sweeps ending anywhere 10kHz-75kHz
    # This is well below even the lowest sweep end, so it will catch the drop
    highpass_freq = min(5000, Fs * 0.4)  # 5kHz or 40% of Nyquist, whichever is lower
    sos = butter(4, highpass_freq, btype='high', fs=Fs, output='sos')
    filtered = sosfiltfilt(sos, signal_window)
    
    # Hilbert envelope - much cleaner than rectification
    envelope = np.abs(hilbert(filtered))
    
    # Fast smoothing with smaller window for better time resolution
    window_size = int(0.01 * Fs)  # 10ms window
    envelope_smooth = uniform_filter1d(envelope, size=window_size, mode='nearest')
    
    # Normalize
    envelope_norm = envelope_smooth / np.max(envelope_smooth)
    
    # Find where envelope drops below threshold
    below_threshold = envelope_norm < threshold
    
    # Find first sustained drop (to avoid false triggers on transients)
    min_samples = int(0.05 * Fs)  # Must stay below for 50ms
    
    # Fast vectorized method using diff to find transitions
    if len(below_threshold) >= min_samples:
        # Find transitions in/out of low region
        padded = np.concatenate(([False], below_threshold, [False]))
        diff = np.diff(padded.astype(int))
        starts = np.where(diff == 1)[0]  # Start of low regions
        ends = np.where(diff == -1)[0]   # End of low regions
        
        # Find first region that's >= min_samples long
        if len(starts) > 0 and len(ends) > 0:
            durations = ends - starts
            long_enough = np.where(durations >= min_samples)[0]
            
            if len(long_enough) > 0:
                end_sample = sample_offset_start + starts[long_enough[0]]
            else:
                # No sustained region, use first drop
                end_sample = sample_offset_start + starts[0]
        else:
            # No low regions at all
            end_sample = sample_offset_end
    else:
        # Window too small, use midpoint
        end_sample = (sample_offset_start + sample_offset_end) // 2
    
    logger.debug(f"End Sample (Global Index): {end_sample}")
    
    if logging.getLogger(__name__).isEnabledFor(logging.DEBUG):
        plot_signal(
            signal_window,
            Fs,
            normalized_signal=envelope_norm,
            threshold=threshold,
            detected_end_time=(end_sample - sample_offset_start) / Fs,
            title="Hilbert Envelope Sweep End Detection",
        )
    
    return end_sample


# Main Processing Function
def slice_audio(input_file, test_record, save_files):
    # Test record parameters
    record_params = {
        'TRS1007': {'sweep_offset': 78, 'sweep_end_min': 48, 'sweep_end_max': 52, 'sweep_start_detect': 0},
        'TRS1005': {'sweep_offset': 32, 'sweep_end_min': 26, 'sweep_end_max': 34, 'sweep_start_detect': 1},
        'STR100': {'sweep_offset': 74, 'sweep_end_min': 63, 'sweep_end_max': 67, 'sweep_start_detect': 0},
        'STR120': {'sweep_offset': 56, 'sweep_end_min': 45, 'sweep_end_max': 50, 'sweep_start_detect': 0},
        'STR130': {'sweep_offset': 80, 'sweep_end_min': 63, 'sweep_end_max': 67, 'sweep_start_detect': 0},
        'STR170': {'sweep_offset': 72, 'sweep_end_min': 63, 'sweep_end_max': 67, 'sweep_start_detect': 0},
        'QR2009': {'sweep_offset': 78, 'sweep_end_min': 48, 'sweep_end_max': 52, 'sweep_start_detect': 0},
        'QR2010': {'sweep_offset': 22, 'sweep_end_min': 15, 'sweep_end_max': 18, 'sweep_start_detect': 0},
        'XG7001': {'sweep_offset': 78, 'sweep_end_min': 48, 'sweep_end_max': 52, 'sweep_start_detect': 0},
        'XG7002': {'sweep_offset': 63, 'sweep_end_min': 26, 'sweep_end_max': 30, 'sweep_start_detect': 1},
        'XG7005': {'sweep_offset': 76, 'sweep_end_min': 48, 'sweep_end_max': 52, 'sweep_start_detect': 0},
        'DIN45543': {'sweep_offset': 78, 'sweep_end_min': 48, 'sweep_end_max': 52, 'sweep_start_detect': 0},
        'ИЗМ33С0327': {'sweep_offset': 58, 'sweep_end_min': 48, 'sweep_end_max': 52, 'sweep_start_detect': 0},

    }

    if test_record.upper() not in record_params:
        raise ValueError("Invalid test record.")

    params = record_params[test_record.upper()]

    # Read input file
    left, right, Fs = read_measurement(input_file)

    logger.info(f"Test Record: {test_record}")

    # Find end of left pilot tone / start of sweep using Hilbert method
    # Note: find_burst_bounds_hilbert does its own filtering
    _, start_left_sweep = find_burst_bounds(left, Fs, tone_freq=1000, threshold=0.3)

    if params['sweep_start_detect'] == 1:
        # For test records where sweep starts several seconds after pilot ends
        # Look for energy rise at 1kHz (sweep start), not another sustained tone
        sample_offset = start_left_sweep + Fs  # Start searching 1s after pilot ends
        start_left_sweep = sample_offset + find_sweep_start(left[sample_offset:], Fs, search_duration=10.0, threshold=0.2)

    logger.info(f"Start of Left Sweep: {start_left_sweep}")

    # Find end of right pilot tone / start of sweep
    sample_offset = start_left_sweep + int(Fs * params['sweep_offset'])
    _, start_right_sweep = sample_offset + find_burst_bounds(right[sample_offset:], Fs, tone_freq=1000, threshold=0.3)

    if params['sweep_start_detect'] == 1:
        # Same for right channel
        sample_offset = start_right_sweep + Fs
        start_right_sweep = sample_offset + find_sweep_start(right[sample_offset:], Fs, search_duration=10.0, threshold=0.2)

    logger.info(f"Start of Right Sweep: {start_right_sweep}")

    # Find end of left sweep using Hilbert method
    # Automatically detects end for sweeps ending 10kHz-75kHz
    end_left_sweep = find_end_of_sweep(
        start_left_sweep, 
        params['sweep_end_min'], 
        params['sweep_end_max'], 
        left,
        Fs
    )
    logger.info(f"End of Left Sweep: {end_left_sweep}")

    # Find end of right sweep
    end_right_sweep = find_end_of_sweep(
        start_right_sweep, 
        params['sweep_end_min'], 
        params['sweep_end_max'], 
        right,
        Fs
    )
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
