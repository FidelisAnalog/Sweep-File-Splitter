# Sweep File Splitter

## Overview
This script processes a stereo audio file (WAV format) to detect and extract sweep segments from specific test records.  This stand-alone version has debugging featuers that the version to be integrated with SJPlot will not. 

## Features
- Reads stereo WAV files and processes both left and right channels.
- Detects pilot tones and sweep signals using customizable parameters.
- Applies bandpass and high-pass filters to isolate signal regions of interest.
- Visualizes signals with optional plots for debugging and analysis.
- Extracts and saves the detected segments into separate output WAV files.

## Requirements
The script requires the following Python libraries:

- numpy
- scipy
- matplotlib

You can install them using pip:

```bash
pip install numpy scipy matplotlib
```

## Usage
Input Parameters
 - INPUT_FILE: Path to the stereo WAV file to be processed.
 - TEST_RECORD: Specifies the test record type. Options are:
    - TRS1007
    - TRS1005
    - STR100

Each test record type adjusts parameters such as sweep offset and detection ranges.

## Running the Script
Run the script using Python:

```bash
python script_name.py
```

Replace script_name.py with the actual name of the script file.

## Output Files
The script generates two output files:

- <input_file>_L.wav: Extracted segment for the left channel.
- <input_file>_R.wav: Extracted segment for the right channel.

## Customization
The script allows customization of the following parameters:

- Filter configurations (low, high, order, etc.).
- Peak detection thresholds and burst boundaries.
- Test record-specific parameters in the record_params dictionary.

## Debugging and Visualization
Signal processing steps can be visualized with plots by enabling logging at the DEBUG level.
To enable detailed logging, modify the logger configuration in the script:

```bash
logger.setLevel(logging.DEBUG)
```

### Burst Detection Visualization
Where: find_burst_bounds

This plot shows the process of identifying a "burst" within the signal, typically corresponding to a specific pattern or tone.

Key Features:
- Peaks: Displays the detected peaks in the signal to ensure proper burst detection.
- Threshold: Highlights the value used to determine significant peaks.
- Detected Start and End Times: Vertical lines marking the burst's start and end points.

Purpose: This helps confirm that the burst detection algorithm correctly identifies the expected signal region.

#### Examples

<br/>
<div align="center" style="padding: 20px 0;">
    <img src="images/Figure_1.png" alt="Example Plot.">
    <p><b>Figure 1 - First Burst Detection Pass</b></p>
    <p>Used to find the end of the first 1kHz pilot tone. In most cases this is also the start of the sweep.</p>
</div>
<br/>

### Sweep End Detection Visualization
Where: find_end_of_sweep

This visualization shows the segment of the signal where the end of a sweep is detected. The goal is to identify the precise sample where the signal drops below a threshold.

Key Features:
- Filtered Signal: The processed signal used for detecting the sweep's end.
- Threshold: A horizontal line marking the detection threshold.
- Detected End Time: A vertical line indicating the sample where the sweep ends.

Purpose: This visualization ensures the end-of-sweep detection algorithm identifies the correct time based on the threshold and signal behavior.

### Segment Visualization
Where: At the end of slice_audio

This shows the final extracted segments (sweeps) for both the left and right channels.

Key Features:
- Segment Plot: Displays the portion of the signal identified as the sweep.
- Title: Indicates whether the plot corresponds to the left or right channel.

Purpose: This visualization allows you to verify that the extracted segments correspond to the expected sweep regions.


### Example
Process the WAV file example.wav with the TRS1007 test record:

INPUT_FILE = 'example.wav'
TEST_RECORD = 'TRS1007'
Run the script, and the output WAV files will be saved in the same directory.

## Contributing
Contributions to improve the script are welcome. Please feel free to fork the repository, make your changes, and submit a pull request.

## License
This project is licensed under the MIT License - see the LICENSE file for details.
