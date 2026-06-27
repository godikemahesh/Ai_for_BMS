# Ai_BMS

Ai_BMS is a compact research codebase for battery management system (BMS) prognosis and state estimation. It combines an Extended Kalman Filter (EKF) approach with a temporal convolutional neural network (Temporal CNN) to produce State of Health (SOH) estimates and Remaining Useful Life (RUL) predictions. A pretrained model is included for inference and evaluation.

## Table of contents
- [Key ideas](#key-ideas)
- [Repository structure](#repository-structure)
- [Requirements & setup](#requirements--setup)
- [Usage examples](#usage-examples)
- [Model files](#model-files)
- [Development notes](#development-notes)
- [Troubleshooting & tips](#troubleshooting--tips)
- [Contributing & contact](#contributing--contact)

## Key ideas
- Combines physics-inspired filtering (EKF) with data-driven temporal-CNN modeling to improve robustness of SOH and RUL estimation.
- Designed for quick inference with a pretrained model included; scripts for prediction and visualization are provided.

## Repository structure

- [AiforBMS/main.py](AiforBMS/main.py) — entry point / example runner.
- [AiforBMS/rul_predictor.py](AiforBMS/rul_predictor.py) — RUL prediction utilities and inference wrapper.
- [AiforBMS/soh_model.py](AiforBMS/soh_model.py) — SOH model definition and inference helpers.
- [AiforBMS/visualization.py](AiforBMS/visualization.py) — plotting and result visualization tools.
- [AiforBMS/requirements.txt](AiforBMS/requirements.txt) — Python dependencies used by the project.
- [ekf_temporal_cnn_model.pth](ekf_temporal_cnn_model.pth) — pretrained PyTorch model (root copy).
- [AiforBMS/ekf_temporal_cnn_model.pth](AiforBMS/ekf_temporal_cnn_model.pth) — pretrained model located inside the package (duplicate copy).

## Requirements & setup
- Python 3.8+ is recommended.
- Install dependencies from the provided requirements file.

PowerShell example:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r AiforBMS/requirements.txt
```

Or with bash / WSL:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r AiforBMS/requirements.txt
```

## Usage examples

These scripts are small wrappers for loading the provided model and running inference. Exact runtime arguments are defined in each script; the following examples show common ways to run them.

- Run the main example/entry script:

```powershell
python AiforBMS/main.py
```

- Run RUL prediction (loads model and runs inference):

```powershell
python AiforBMS/rul_predictor.py
```

- Run SOH inference or utilities:

```powershell
python AiforBMS/soh_model.py
```

- Produce plots from prediction outputs:

```powershell
python AiforBMS/visualization.py
```

If a script requires input data (CSV or serialized arrays), inspect the top of the script for argument parsing and expected formats. Typical input is time-series battery telemetry such as voltage, current, temperature, and timestamps.

## Model files
- The pretrained model is included as [ekf_temporal_cnn_model.pth](ekf_temporal_cnn_model.pth). If your scripts expect the model inside the package, copy or symlink it to [AiforBMS/ekf_temporal_cnn_model.pth](AiforBMS/ekf_temporal_cnn_model.pth).
- The model is a PyTorch state dict — load with `torch.load(..., map_location=device)` and `model.load_state_dict(...)` before `model.eval()`.

## Development notes
- Code is organized as small, focused scripts. To adapt the pipeline:
	- Modify data-preprocessing inside `AiforBMS/rul_predictor.py` or `AiforBMS/soh_model.py` to match your dataset columns.
	- Retrain or fine-tune the Temporal CNN by adding a training loop: the repository currently includes inference-ready code and a pretrained snapshot.
- For GPU inference set `device = torch.device('cuda')` when available.

## Troubleshooting & tips
- If you see CUDA-related errors on a machine without a GPU, ensure you set `map_location='cpu'` when loading the model.
- If a script fails due to missing packages, double-check `AiforBMS/requirements.txt` and rerun `pip install -r AiforBMS/requirements.txt`.
- To inspect expected input shapes and column ordering, open the preprocessing section at the top of `AiforBMS/rul_predictor.py`.

## Contributing & contact
- Contributions welcome: open an issue or a pull request describing the change and tests/data needed.
- No license file present in this repository. Add a `LICENSE` file if you intend to make the project open source, or contact the repository owner for permission before reuse.

---

## Detailed explanation (for beginners)

This section explains the key battery management concepts, the sensors and signals commonly used, and how this project applies AI together with an EKF to estimate battery health and life. No prior specialist knowledge is required.

### What is a Battery Management System (BMS)?
- The BMS monitors and protects rechargeable batteries (cells and packs). Its goals include:
	- Safety: detect unsafe conditions (over-voltage, over-current, over-temperature).
	- Monitoring: measure and estimate internal battery states that cannot be read directly (for example, State of Charge).
	- Prognosis: predict how long the battery will keep working (Remaining Useful Life).

### Core battery state definitions
- State of Charge (SOC): a percentage expressing the available charge relative to capacity (0% empty, 100% full). SOC is like a fuel gauge for batteries.
- State of Health (SOH): a measure of the battery's condition compared to a new battery (usually expressed as a percent of original capacity). SOH decreases over calendar and cycle aging.
- Remaining Useful Life (RUL): an estimate of how long (or how many cycles) the battery will continue to meet a defined end-of-life criterion (for example, SOH falling below 70%).

### What sensors and signals are used?
- Voltage: measures terminal voltage of a cell or pack. Voltage response to load and rest provides information about internal resistance and capacity.
- Current: the charging/discharging current. Integrating current over time (Coulomb counting) helps estimate SOC; current also drives internal heating and degradation.
- Temperature: battery temperature strongly affects capacity, internal resistance, and degradation rate. Temperature spikes can indicate problems.

In practice, the BMS records sequences of voltage, current, and temperature over time. These sequences are the inputs to models that estimate SOC, SOH, and RUL.

### Why combine EKF and AI (Temporal CNN)?
- Extended Kalman Filter (EKF): a physics-inspired estimator that uses an internal electrochemical/empirical model and new measurements to produce filtered estimates of hidden states (e.g., SOC) while respecting known dynamics and uncertainty. EKF provides smoothing, interpretable state estimates, and uncertainty modeling.
- Temporal CNN (data-driven): a neural network that processes time-series windows (sequences of recent voltage/current/temperature) and learns complex mappings to targets like SOH or RUL from data. Temporal CNNs can efficiently learn temporal patterns and trends in sensor signals.
- Combined approach (used in this project): the EKF offers physically grounded estimates and features; the Temporal CNN learns residual patterns and longer-term degradation signals. Combining them improves robustness: the EKF maintains physical consistency while the CNN captures empirical aging patterns the EKF can't model alone.

### How this project uses signals to produce SOH, RUL and SOC
- Preprocessing: raw telemetry (voltage, current, temperature, timestamps) is resampled/normalized and segmented into short time windows (e.g., last N timesteps). Missing values are handled by interpolation or masking.
- EKF step: for each step/window the EKF uses a simple battery model and the latest measurements to estimate dynamic states such as SOC and internal states (and their uncertainties). These EKF outputs are saved as per-timestep features.
- Feature construction: the pipeline concatenates raw measurements (voltage/current/temperature), hand-crafted features (for example, moving averages, discharge capacity from Coulomb counting), and EKF outputs (filtered SOC, internal estimates, uncertainties) into a per-window feature tensor.
- Temporal CNN inference: the CNN consumes the feature tensor for a sequence window and outputs predictions such as SOH (percentage) and RUL (time or cycles until end-of-life). The CNN is trained to minimize regression loss (MAE/MSE) on labeled historic degradation trajectories.

Notes:
- SOC: in this repository SOC is primarily provided by the EKF estimate (Coulomb counting corrected by EKF and observations). The CNN can refine SOC estimates when trained to do so.
- SOH & RUL: these are the primary learning targets of the Temporal CNN. SOH is typically learned as a continuous value (% of original capacity). RUL can be learned directly as time-to-failure or converted from predicted SOH trajectories.

### Typical data format (example CSV)

timestamp, voltage_V, current_A, temperature_C
2021-01-01T00:00:00, 3.76, -0.5, 25.1
2021-01-01T00:00:10, 3.74, -0.7, 25.3

- `timestamp`: ISO8601 or epoch seconds
- `voltage_V`: terminal voltage in volts
- `current_A`: positive for charging, negative for discharging (or vice versa; be consistent)
- `temperature_C`: cell/pack temperature in Celsius

When preparing training data you also need a target label per sample or cycle: measured capacity (for computing SOH) and an EOL threshold to compute RUL.

### How to load the pretrained model (PyTorch)

```python
import torch
from AiforBMS.soh_model import YourModelClass  # adapt to actual class name in repo

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
state = torch.load('AiforBMS/ekf_temporal_cnn_model.pth', map_location=device)
model = YourModelClass(...)
model.load_state_dict(state)
model.to(device)
model.eval()
```

Replace `YourModelClass` with the model class defined in `AiforBMS/soh_model.py` (or wherever the network is implemented). If the repository stores a `state_dict`, use `load_state_dict`. If it saved an entire model object, `torch.load` may return the model directly.

### Training and evaluation (high-level)
- Loss functions: commonly Mean Absolute Error (MAE) or Mean Squared Error (MSE) for SOH/RUL regression.
- Metrics: MAE, RMSE, and time-to-failure classification accuracy (if RUL is discretized).
- Validation: hold out cell trajectories or cycles to avoid leakage between training and test sets (test on cells not seen during training).

### Practical tips
- Use `map_location='cpu'` if you don't have a GPU to avoid CUDA errors.
- Normalize inputs per-feature (zero mean, unit variance) using statistics from the training set. Store normalizers for inference.
- If your currents are cumulative or use different sign conventions, standardize to a single convention and document it.

## Where to look in this repository
- `AiforBMS/main.py` — example script that runs an end-to-end flow (load data → preprocess → infer → visualize).
- `AiforBMS/rul_predictor.py` — contains routines that wrap model inference for RUL.
- `AiforBMS/soh_model.py` — neural network and model utilities.
- `AiforBMS/visualization.py` — helper functions to plot predictions versus ground truth.

## Next steps I can take for you
- Add a small, synthetic demo dataset and a minimal notebook showing preprocessing and inference.
- Add explicit CLI flags to `AiforBMS/main.py` and documented examples.
- Run an inference pass on a sample CSV you provide and return plots and metric outputs.

If you'd like any of the above, tell me which one and I'll implement it.

