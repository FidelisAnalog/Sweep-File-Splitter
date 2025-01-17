# Sweep File Splitter

## Overview
This script processes a stereo audio file (WAV format) to detect and extract sweep segments from specific test records.  This stand-alone version has debugging featuers that the version integrated with SJPlot does not. 

Much thanks to DrCWO for contrbuting the Scilab code this is based on. 

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
Input parameters need to be edited in the "# User Parameters" section of the script:
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
    <img src="images/Figure_1.png" alt="Figure 1 - First Burst Detection Pass: Left Channel">
    <p><b>Figure 1 - First Burst Detection Pass: Left Channel</b></p>
    <p>Used to find the end of the first 1kHz pilot tone. In most cases this is also the start of the sweep.</p>
</div>
<br/>

<br/>
<div align="center" style="padding: 20px 0;">
    <img src="images/Figure_2.png" alt="Figure 2 - Optional Sweep Start Detection Pass: Left Channel">
    <p><b>Figure 2 - Optional Sweep Start Detection Pass: Left Channel</b></p>
    <p>In cases like JVC TRS-1005 were there is silence between the end of the pilot tone and the start of the sweep, this is used to find the start of the sweep. This detection window begins one second after the detected end of the pilot tone.</p>
</div>
<br/>

<br/>
<div align="center" style="padding: 20px 0;">
    <img src="images/Figure_3.png" alt="Figure 3 - Second Burst Detection Pass: Right Channel">
    <p><b>Figure 3 - Second Burst Detection Pass: Right Channel</b></p>
    <p>Used to find the end of the second 1kHz pilot tone. In most cases this is also the start of the sweep. The start of this detection window is however many seconds "sweep_offset" is configured for from the end of the first pilot tone.</p>
</div>
<br/>

<br/>
<div align="center" style="padding: 20px 0;">
    <img src="images/Figure_4.png" alt="Figure 4 - Optional Sweep Start Detection Pass: Right Channel">
    <p><b>Figure 4 - Optional Sweep Start Detection Pass: Right Channel</b></p>
    <p>In cases like JVC TRS-1005 were there is silence between the end of the pilot tone and the start of the sweep, this is used to find the start of the sweep. This detection window begins one second after the detected end of the second pilot tone.</p>
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

#### Examples
<br/>
<div align="center" style="padding: 20px 0;">
    <img src="images/Figure_5.png" alt="Figure 5 - First End of Sweep Detection Pass: Left Channel">
    <p><b>Figure 5 - First End of Sweep Detection Pass: Left Channel</b></p>
    <p>Used to find the end of the left channel sweep.  The distance and duration of this detection window are set by the variables "sweep_end_min" and "sweep_end_max".</p>
</div>
<br/>

<br/>
<div align="center" style="padding: 20px 0;">
    <img src="images/Figure_6.png" alt="Figure 6 - First End of Sweep Detection Pass: Right Channel">
    <p><b>Figure 6 - First End of Sweep Detection Pass: Right Channel</b></p>
    <p>Used to find the end of the right channel sweep.  The distance and duration of this detection window are set by the variables "sweep_end_min" and "sweep_end_max".</p>
</div>
<br/>

### Segment Visualization
Where: At the end of slice_audio

This shows the final extracted segments (sweeps) for both the left and right channels.

Key Features:
- Segment Plot: Displays the portion of the signal identified as the sweep.
- Title: Indicates whether the plot corresponds to the left or right channel.

Purpose: This visualization allows you to verify that the extracted segments correspond to the expected sweep regions.

#### Examples
<br/>
<div align="center" style="padding: 20px 0;">
    <img src="images/Figure_7.png" alt="Figure 7 - Extracted Sweep Segment: Left Channel">
    <p><b>Figure 7 - Extracted Sweep Segment: Left Channel</b></p>
    <p>This is a waveform visualization of the extracted sweep segment for the left channel.</p>
</div>
<br/>

<br/>
<div align="center" style="padding: 20px 0;">
    <img src="images/Figure_8.png" alt="Figure 8 - Extracted Sweep Segment: Right Channel">
    <p><b>Figure 8 - Extracted Sweep Segment: Right Channel</b></p>
    <p>This is a waveform visualization of the extracted sweep segment for the right channel.</p>
</div>
<br/>

## Contributing
Contributions to improve the script are welcome. Please feel free to fork the repository, make your changes, and submit a pull request.

## License
This project is licensed under the MIT License - see the LICENSE file for details.
