# RCBeam3D-FourPointBending canonical archive

This folder holds metadata for the candidate four-point-bending archive. HDF5
is the only benchmark interchange format; the large case files are not copied
into this documentation repository.

## Expected upload shape

When the release list and license are frozen, a self-contained canonical archive
should contain:

```text
RCBeam3D-FourPointBending/
├── cases/
│   └── <case_id>.h5
├── splits/
│   ├── train.txt
│   ├── validation.txt
│   └── test.txt
├── README.md
├── card.json
├── cases.csv
└── LICENSE.txt
```

`cases.csv` records the measured inventory and keeps every split value as
`unassigned` until the exact train/validation/test case-id lists are approved.

## HDF5 conventions

- Length is stored in m.
- Stress is stored in Pa.
- Force is stored in N.
- PEEQ is dimensionless.
- Connectivity stores zero-based row indices.
- Each file contains one complete loading trajectory.

Do not create `LICENSE.txt` by guessing a license; the rights holder must
supply the applicable terms.


Case-id split lists are not bundled in this preview.
