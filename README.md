# Example Analysis

This directory hold analysis example. The analysis may directly interact with DROP or perform analysis on the output RQ files.

- `uproot_example_v1.0.ipynb`: a bare minimal example of how to read our data into python via uproot. The raw data are available in root file. 
- `led.ipynb`: led calibration. We took LED runs at various intensity. This script loads multiple root files, find the intensity best for a PMT, and plot charge distributions.
- `data_quality_monitor`: it monitors muon data with a series of plot

# WbLS Calibration 30T

. calibration30t sumSPE30t_v2.py YYMMDD HHmm

- check directory for path defined in sumSPE30t_v2.py

# WbLS Calibraiton 1T

. repscript.sh sumSPE1t.py list.txt

- check directory
- make list.txt for target days

## Calibration validating
CheckCSVvaildation.ipynb

- first cell run 30t validaton
- second cell run 1t validation : when 1ton b1ch0 issue cleared, remove force setting as 0

## Calibration Monitor graph
. CalibrationMonitor.py

## CheckCalibration.ipynb
When you want to look waveform directly, in range of charge, use this
