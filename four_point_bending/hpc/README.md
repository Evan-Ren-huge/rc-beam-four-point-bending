# Viewing and running the KSH files

KSH files are plain-text shell scripts. They are meant to be read in any text
editor and executed by KornShell on Linux, WSL or an HPC login node.

## Open without running

PowerShell:

```powershell
Get-Content .\prepare_and_validate.ksh
notepad .\prepare_and_validate.ksh
```

VS Code, if the `code` command is installed:

```powershell
code .\prepare_and_validate.ksh
```

## Run under WSL/Linux

```bash
cd /path/to/meiyouyong/rc-beam-four-point-bending
python3 -m pip install -r requirements.txt

NPZ_ROOT=/mnt/d/obd2npz3 \
H5_ROOT=/mnt/d/obd2npz3_canonical_h5 \
ksh hpc/prepare_and_validate.ksh
```

Render the first canonical case:

```bash
H5_ROOT=/mnt/d/obd2npz3_canonical_h5 \
CASE_ID=Job-new0a0b0 \
FIELD=stress \
ksh hpc/render_case.ksh
```

Render the synchronous ground-truth and three-model 3D comparison:

```bash
TRAIN_ROOT=/path/to/model-rollouts \
CASE_ID=Job-new0a0b0 \
FIELD=stress \
ksh hpc/render_comparison_3d.ksh
```

On Windows without WSL/KornShell, run the Python file directly; opening or
double-clicking a `.ksh` file does not execute the Python visualization.


