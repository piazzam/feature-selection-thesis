# Data

Most datasets are fetched automatically from UCI or OpenML by `src/generation.py`.
A small subset is expected as local files under a `DATA/` directory in the repository root, using the original filenames referenced in the code:

- `Hill_Valley_without_noise_Training.data`
- `Hill_Valley_without_noise_Testing.data`
- `Hill_Valley_with_noise_Training.data`
- `Hill_Valley_with_noise_Testing.data`
- `movement_libras.data`
- `gas_sensor_array_drift_dataset/*.dat`
- `PersonGaitDataSet.mat`
- `DIA_trainingset_RDKit_descriptors.csv`
- `LSVT_voice_rehabilitation.xlsx`

These datasets are not redistributed in this repository. Obtain them from their original sources and place them under `DATA/` with the paths above.


For Hill-Valley, the separate testing files are required because the thesis protocol keeps the predefined test partition fixed and uses it only for final evaluation.
