"""Execute the notebook's actual definitions on tiny synthetic CPU batches."""
import ast
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import Dataset, DataLoader

NOTEBOOK = Path(__file__).resolve().parents[1] / "3D_image_classification.ipynb"


def definitions():
    scope = {"torch": torch, "np": np, "Dataset": Dataset}
    notebook = json.loads(NOTEBOOK.read_text())
    for cell in notebook["cells"]:
        source = "".join(cell.get("source", []))
        if cell["cell_type"] != "code" or source.startswith(("!", "%")):
            continue
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue
        nodes = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))
                 and node.name in {"CTScanDataset", "train_one_epoch", "evaluate"}]
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(NOTEBOOK), "exec"), scope)
    return scope


@pytest.mark.parametrize("size", [1, 2])
def test_binary_training_accepts_singleton_and_regular_batches(size):
    scope = definitions()
    scans = [np.zeros((2, 2, 2), dtype=np.float32) for _ in range(size)]
    dataset = scope["CTScanDataset"](scans, [i % 2 for i in range(size)], train=False)
    loader = DataLoader(dataset, batch_size=size)
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(8, 1), torch.nn.Sigmoid())
    optimizer = torch.optim.SGD(model.parameters(), lr=.1)
    criterion = torch.nn.BCELoss()
    loss = scope["train_one_epoch"](model, optimizer, criterion, loader, "cpu")
    val_loss, accuracy = scope["evaluate"](model, criterion, loader, "cpu")
    assert np.isfinite(loss) and np.isfinite(val_loss)
    assert 0 <= accuracy <= 1


def test_notebook_split_is_stratified_disjoint_and_replayable(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    notebook = json.loads(NOTEBOOK.read_text())
    scope = {"np": np, "normal_scan_paths": [f"normal/{i}.nii" for i in range(20)],
             "abnormal_scan_paths": [f"abnormal/{i+20}.nii" for i in range(20)],
             "process_scan": lambda path: np.zeros((2, 2, 2), dtype=np.float32)}
    source = "".join(notebook["cells"][14]["source"])
    exec(compile(source, str(NOTEBOOK), "exec"), scope)
    first = (tmp_path / "split_manifest.json").read_bytes()
    groups = [set(scope[name]) for name in ("train_indices", "val_indices", "test_indices")]
    assert [len(group) for group in groups] == [28, 6, 6]
    assert len(set.union(*groups)) == 40
    assert all(groups[i].isdisjoint(groups[j]) for i in range(3) for j in range(i+1, 3))
    assert all({scope["all_labels"][i] for i in group} == {0, 1} for group in groups)
    exec(compile(source, str(NOTEBOOK), "exec"), scope)
    assert (tmp_path / "split_manifest.json").read_bytes() == first


def test_selected_pytorch_checkpoint_loads_and_evaluates_synthetic_test(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(8, 1), torch.nn.Sigmoid())
    torch.save(model.state_dict(), "best_model.pth")
    scope = definitions()
    scope.update(network=model, device="cpu", DataLoader=DataLoader,
                 all_scans=[np.zeros((2, 2, 2), dtype=np.float32)], all_labels=[0],
                 test_indices=[0], criterion=torch.nn.BCELoss())
    source = "".join(json.loads(NOTEBOOK.read_text())["cells"][41]["source"])
    exec(compile(source, str(NOTEBOOK), "exec"), scope)
    assert np.isfinite(scope["test_loss"])
    assert 0 <= scope["test_accuracy"] <= 1


def test_nifti_affine_orientation_and_index_grid_shape(tmp_path):
    nib = pytest.importorskip("nibabel")
    from scipy import ndimage

    scope = {"nib": nib, "np": np, "ndimage": ndimage}
    source = "".join(json.loads(NOTEBOOK.read_text())["cells"][10]["source"])
    exec(compile(source, str(NOTEBOOK), "exec"), scope)
    data = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    path = tmp_path / "left-handed.nii.gz"
    nib.save(nib.Nifti1Image(data, np.diag([-2., 3., 4., 1.])), path)
    canonical = scope["read_nifti_file"](path)
    np.testing.assert_array_equal(canonical, data[::-1])
    resized = scope["process_scan"](path)
    assert resized.shape == (128, 128, 64)
    assert resized.dtype == np.float32
    assert np.isfinite(resized).all()
    assert resized.min() >= 0 and resized.max() <= 1
