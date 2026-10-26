import numpy as np
import pytest

from src.load_smd import MinMaxScaler, list_machines, load_interpretation, load_machine


@pytest.fixture
def smd_root(tmp_path):
    for split in ("train", "test", "test_label", "interpretation_label"):
        (tmp_path / split).mkdir()
    for name in ("machine-1-2", "machine-1-10", "machine-2-1"):
        np.savetxt(tmp_path / "train" / f"{name}.txt", np.ones((4, 3)), delimiter=",")
        np.savetxt(tmp_path / "test" / f"{name}.txt", np.ones((5, 3)), delimiter=",")
        np.savetxt(tmp_path / "test_label" / f"{name}.txt", [0, 0, 1, 1, 0], fmt="%d")
    (tmp_path / "interpretation_label" / "machine-1-2.txt").write_text("2-3:1,3\n")
    return tmp_path


def test_machines_sort_numerically(smd_root):
    assert list_machines(smd_root) == ["machine-1-2", "machine-1-10", "machine-2-1"]


def test_load_machine_shapes(smd_root):
    m = load_machine("machine-1-2", smd_root)
    assert m.train.shape == (4, 3) and m.test.shape == (5, 3)
    assert m.labels.tolist() == [0, 0, 1, 1, 0]


def test_label_row_mismatch_raises(smd_root):
    np.savetxt(smd_root / "test_label" / "machine-2-1.txt", [0, 1], fmt="%d")
    with pytest.raises(ValueError, match="labels"):
        load_machine("machine-2-1", smd_root)


def test_interpretation_columns_become_zero_indexed(smd_root):
    [seg] = load_interpretation("machine-1-2", smd_root)
    assert (seg.start, seg.end, seg.columns) == (2, 3, (0, 2))


def test_scaler_fits_train_only_and_does_not_clip():
    train = np.array([[0.0, 5.0], [10.0, 5.0]])
    scaler = MinMaxScaler().fit(train)
    out = scaler.transform(np.array([[20.0, 5.0]]))
    assert out[0, 0] == 2.0   # beyond the train range, kept
    assert out[0, 1] == 0.0   # constant metric, no division by zero
