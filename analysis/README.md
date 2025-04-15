# Example Analysis

This directory hold analysis example. The analysis may directly interact with DROP or perform analysis on the output RQ files.

- `uproot_example_v1.0.ipynb`: a bare minimal example of how to read our data into python via uproot. The raw data are available in root file.
- `led.ipynb`: led calibration. We took LED runs at various intensity. This script loads multiple root files, find the intensity best for a PMT, and plot charge distributions.
- `data_quality_monitor`: it monitors muon data with a series of plot


## Calibration process

We have everyday 30t calibration dataset file, named as 'majority_test_YYMMDDTHHmm'.
- Check where dataset location is, and change diretory at sumSPE30t_v2.py
- run calibration with . calibration30t.sh sumSPE30t_v2.py YYMMDD HHmm
- after calbration done, every few days (in my case it was a week) validate CSV file with "CheckCSVvalidation.ipynb"
- If you want to see each waveform in some region of charge, use "CheckCalibration.ipynb"

For 1ton,
- Check Dataset location for sumSPE1t.py
- make "list.txt" that has target calibration day list like 250415 250146 ...
- run ". repscript.sh sumSPE1t.py"
