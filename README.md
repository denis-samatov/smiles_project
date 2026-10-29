# smiles_project

Exploratory 3D CNN for classifying viral pneumonia (COVID-19) from chest CT scans.

- **Data:** [MosMedData: Chest CT Scans with COVID-19 Related Findings](https://www.medrxiv.org/content/10.1101/2020.05.20.20100362v1) (public subset).
- **Task:** binary classification — CT scan shows signs of viral pneumonia, or is normal.
- **Approach:** a PyTorch 3D CNN trained on a small (200-scan) subset, plus a setup step for
  [MedNeXt](https://github.com/MIC-DKFZ/MedNeXt) as an alternative 3D architecture.
- **Notebook:** [`3D_image_classification.ipynb`](3D_image_classification.ipynb).

This is an adaptation of Hasib Zunair's [3D image classification tutorial](https://github.com/hasibzunair/3D-image-classification-tutorial)
(ported from Keras to PyTorch), not original architecture work. On the full ~1,000-scan
dataset the source tutorial reports ~83% accuracy with 6-7% run-to-run variance; this
repository's own run used only the 200-scan subset with no fixed seed, so its result is
not a reliable estimate — treat this as an exploratory notebook, not a validated model.

## Related repositories

- [`radiomics`](https://github.com/denis-samatov/radiomics) — the author's other medical-imaging
  work (myocardial radiomic feature extraction), unrelated task and dataset.
- [`smiles`](https://github.com/denis-samatov/smiles) — an unrelated repository (vision-language
  model guides); the similar name is coincidental.


## Corrected software path and validation boundary

Run the notebook in order from a fresh kernel. Targets are float binary labels; batch size 1
is supported. Training, history and checkpoint inference now consistently use PyTorch.
The small MedNeXt default is for memory-bounded development, not the historical configuration.
Outputs of changed model/training/inference cells were cleared; no replacement performance
numbers or fresh CT training are claimed. Historical unrelated data-loading output is retained.

The seed and sorted scan paths fix a stratified 70/15/15 **scan-level** split, saved locally as
`split_manifest.json`. Validation selects checkpoints; test is held out until final evaluation.
Without patient/site identifiers this does not establish patient-disjoint evaluation.
NIfTI is canonicalized using its affine before HU clipping and resizing to 128×128×64. The final
array resize is an index-grid model input convention; physical spacing/distances are not
preserved and the model must not be interpreted as a physical measurement tool.

Offline CPU regression (synthetic 2×2×2 volumes; no download):

```bash
python -m pytest tests -q
```

Before a new scientific run, pin the MedNeXt source revision and environment, record source
archive SHA-256 plus rights/attribution, check patient grouping, freeze the split manifest,
measure actual CPU/GPU peak memory with batch 1, and compare a simple baseline. Save test
predictions and report sensitivity/specificity/AUROC with uncertainty and calibration limits.
The two historical CT archives/model weights are not downloaded or qualified by these tests.
Repository availability and these local repairs do not establish clinical or production readiness.

The additional synthetic NIfTI test checks affine-driven left/right canonicalization and the
128×128×64 index-grid output using NiBabel 5.4.2. It does not qualify physical spacing fidelity,
full MedNeXt peak memory or a real CT population.
