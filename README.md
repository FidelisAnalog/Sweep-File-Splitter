# Sweep File Splitter

## Overview
This script processes a stereo audio file (WAV format) to detect and extract specific segments of interest, such as pilot tones and sweep signals, using filtering, peak detection, and signal normalization.

## Features
- Reads stereo WAV files and processes both left and right channels.
- Detects pilot tones and sweep signals using customizable parameters.
- Applies bandpass and high-pass filters to isolate signal regions of interest.
- Visualizes signals with optional plots for debugging and analysis.
- Extracts and saves the detected segments into separate output WAV files.
- Requirements


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

### Example

Process the WAV file example.wav with the TRS1007 test record:

INPUT_FILE = 'example.wav'
TEST_RECORD = 'TRS1007'
Run the script, and the output WAV files will be saved in the same directory.

## Contributing
Contributions to improve the script are welcome. Please feel free to fork the repository, make your changes, and submit a pull request.

## License
This project is licensed under the MIT License - see the LICENSE file for details.
