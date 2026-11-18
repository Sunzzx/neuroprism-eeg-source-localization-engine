# Neuroprism EEG Source Localization Engine

3-sphere leadfield forward model + ECD / MNE / sLORETA inverse solvers for EEG.

## Install

```bash
pip install -e .
pytest -q
```

## Example

```python
from neuroprism import simulate_eeg_from_dipole, fit_equivalent_current_dipole, inverse_sloreta
import numpy as np

pos = np.array([0.02, 0.0, 0.045])
m = np.array([0.0, 0.0, 5e-7])
V, electrodes, labels = simulate_eeg_from_dipole(pos, m)
fit = fit_equivalent_current_dipole(V, electrodes)
print(fit.position, fit.gof)
```
