# Extracted data schema

[Back to dataset card](../README.md)

Source: Abaqus extraction for `HolePlate2D-Damage-800`, combining 100 + 200 + 500 non-overlapping full-edge cases. One `<case_id>_damage_gnn.npz` plus a JSON summary per case. Shapes below are read from the representative case. `T=101`, `N=1684`, `E=1580`.

| NPZ key | Shape | Stored dtype |
|---|---|---|
| `case_id` | `()` | `<U18` |
| `geometry_json` | `()` | `<U1864` |
| `normalized_progress` | `(101,)` | `float64` |
| `prescribed_displacement_history_mm` | `(101,)` | `float64` |
| `node_labels` | `(1684,)` | `int64` |
| `node_coordinates` | `(1684, 2)` | `float32` |
| `fixed_node_mask` | `(1684,)` | `bool` |
| `right_boundary_node_mask` | `(1684,)` | `bool` |
| `loaded_node_mask` | `(1684,)` | `bool` |
| `boundary_code` | `(1684, 1)` | `float64` |
| `element_labels` | `(1580,)` | `int64` |
| `connectivity` | `(1580, 4)` | `int64` |
| `nodal_displacement_history` | `(101, 1684, 2)` | `float64` |
| `integration_point_stress_components_history_mpa` | `(101, 1580, 4)` | `float64` |
| `integration_point_mises_history_mpa` | `(101, 1580)` | `float64` |
| `integration_point_peeq_history` | `(101, 1580)` | `float64` |
| `integration_point_damage_history` | `(101, 1580)` | `float64` |
| `element_status_history` | `(101, 1580)` | `float64` |
| `total_reaction_force_history_n` | `(101,)` | `float64` |
| `tensile_force_magnitude_history_n` | `(101,)` | `float64` |
| `allie_history_n_mm` | `(101,)` | `float64` |
| `allae_history_n_mm` | `(101,)` | `float64` |
| `allwk_history_n_mm` | `(101,)` | `float64` |
| `allpd_history_n_mm` | `(101,)` | `float64` |
| `allsd_history_n_mm` | `(101,)` | `float64` |

## Indexing and units

- `connectivity` contains Abaqus node labels, not zero-based array positions. Map through `node_labels` before indexing coordinates. Element histories follow `element_labels`.
- Coordinates, prescribed displacement and nodal U1/U2 use mm; stress and von Mises use MPa; reaction uses N; energy histories use N·mm.
- PEEQ, SDEG and STATUS are dimensionless. STATUS=1 is active; STATUS=0 is deleted.
- The stress array preserves the four-component order returned by the Abaqus plane-stress S field. Component labels are not saved in NPZ; confirm their order from the source ODB before using individual tensor components. The saved von Mises scalar is exported directly from `value.mises`.
- `normalized_progress` holds the saved step-frame values over a unit-duration monotonic ramp. This is a quasi-static loading coordinate, not animation playback time.
- `geometry_json` is a JSON string containing geometry and generation metadata.

## Extraction conventions

The extractor initializes missing stress values to zero, carries the last PEEQ value forward when unavailable, and defaults missing SDEG values to 1 for deleted elements or 0 for active elements when the SDEG field exists. Use STATUS explicitly; zeros in stress after deletion are not measurements of an intact material. History outputs (force, prescribed displacement and energy) are interpolated to the saved field-frame times.

## Elasticity and hardening

E = 210000 MPa; Poisson ratio = 0.30. Isotropic plastic hardening:

| Yield stress [MPa] | Equivalent plastic strain |
|---:|---:|
| 250 | 0 |
| 270 | 0.01 |
| 295 | 0.03 |
| 320 | 0.06 |
| 345 | 0.1 |
| 370 | 0.16 |
| 390 | 0.22 |

## Ductile damage initiation

| Equivalent plastic strain at initiation | Stress triaxiality | Strain rate |
|---:|---:|---:|
| 0.35 | -0.333333 | 0 |
| 0.25 | 0 | 0 |
| 0.15 | 0.333333 | 0 |
| 0.1 | 0.666667 | 0 |

Linear displacement-based damage evolution; failure displacement parameter 0.25 mm. Synthetic baseline, not alloy-specific calibration.

## Analysis

Abaqus/Standard Static, General with `NLGEOM=YES`. Initial stabilization fraction 0.0002. The source summaries retain solver warnings and stabilization-energy diagnostics. Consult these diagnostics together with the batch audit when selecting cases for training.

## Safe loading example

```python
import numpy as np
with np.load("<case_id>_damage_gnn.npz", allow_pickle=False) as data:
    xyz = data["node_coordinates"]
    label_to_row = {int(label): i for i, label in enumerate(data["node_labels"])}
    conn = np.array([[label_to_row[int(n)] for n in e] for e in data["connectivity"]])
    displacement = data["nodal_displacement_history"]
    magnitude_mm = np.linalg.norm(displacement, axis=-1)
    active = data["element_status_history"] > 0.5
```
