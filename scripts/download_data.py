"""Download the public benchmark datasets into data/ and verify them.

SMD (Server Machine Dataset) comes from a sparse checkout of the OmniAnomaly
repository. RCAEval suites come from Zenodo record 14590730; every archive is
checked against the MD5 that Zenodo publishes before it is extracted.

Re-running is safe: anything already downloaded and verified is skipped, and
an interrupted archive resumes where it stopped.

    python scripts/download_data.py                 # SMD + RE1
    python scripts/download_data.py --rcaeval RE2   # one more suite
    python scripts/download_data.py --verify-only   # checks, no downloads
"""

import argparse
import hashlib
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import numpy as np
import requests

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
SMD_DIR = DATA / "smd"
RCAEVAL_DIR = DATA / "rcaeval"

OMNI_REPO = "https://github.com/NetManAIOps/OmniAnomaly.git"
ZENODO_RECORD = "https://zenodo.org/api/records/14590730"
SYSTEMS = ("OB", "SS", "TT")  # Online Boutique, Sock Shop, Train Ticket
COMPLETE_MARKER = ".complete"

RCAEVAL_CASES = {"RE1": 375, "RE2": 270, "RE3": 90}

SMD_SPLITS = ("train", "test", "test_label", "interpretation_label")
SMD_MACHINES = 28
SMD_METRICS = 38


def download_smd():
    if (SMD_DIR / COMPLETE_MARKER).exists():
        print("SMD: already downloaded")
        return
    print("SMD: sparse checkout of OmniAnomaly/ServerMachineDataset")
    with tempfile.TemporaryDirectory(dir=DATA) as tmp:
        clone = Path(tmp) / "omni"
        subprocess.run(
            ["git", "clone", "--quiet", "--filter=blob:none", "--sparse", OMNI_REPO, str(clone)],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(clone), "sparse-checkout", "set", "ServerMachineDataset"],
            check=True,
        )
        if SMD_DIR.exists():
            shutil.rmtree(SMD_DIR)
        shutil.copytree(clone / "ServerMachineDataset", SMD_DIR)
    (SMD_DIR / COMPLETE_MARKER).touch()


def zenodo_files():
    response = requests.get(ZENODO_RECORD, timeout=30)
    response.raise_for_status()
    return {f["key"]: f for f in response.json()["files"]}


def fetch(url, dest, expected_md5, size, attempts=10):
    """Download with HTTP Range resume; Zenodo drops long transfers mid-file."""
    next_report = 0.0
    for attempt in range(1, attempts + 1):
        have = dest.stat().st_size if dest.exists() else 0
        if have >= size:
            break
        headers = {"Range": f"bytes={have}-"} if have else {}
        try:
            with requests.get(url, stream=True, timeout=60, headers=headers) as response:
                response.raise_for_status()
                if have and response.status_code != 206:  # server ignored the range
                    have = 0
                with open(dest, "ab" if have else "wb") as out:
                    for chunk in response.iter_content(chunk_size=1 << 20):
                        out.write(chunk)
                        have += len(chunk)
                        if have / size >= next_report:
                            print(f"  {dest.name}: {have / size:5.0%} of {size / 1e6:,.0f} MB", flush=True)
                            next_report = have / size + 0.10
        except (requests.ConnectionError, requests.Timeout) as error:
            print(f"  {dest.name}: interrupted at {have / 1e6:,.0f} MB ({type(error).__name__}), "
                  f"retry {attempt}/{attempts}", flush=True)
            time.sleep(min(60, 5 * attempt))
    else:
        raise RuntimeError(f"{dest.name}: incomplete after {attempts} attempts")

    md5 = hashlib.md5()
    with open(dest, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            md5.update(chunk)
    if md5.hexdigest() != expected_md5:
        dest.unlink()
        raise RuntimeError(f"{dest.name}: MD5 mismatch, download discarded")


def safe_extract(archive, target):
    """Extract a zip, refusing any member that would land outside target."""
    target = target.resolve()
    with zipfile.ZipFile(archive) as zf:
        for member in zf.namelist():
            if not (target / member).resolve().is_relative_to(target):
                raise RuntimeError(f"{archive.name}: unsafe path {member!r}")
        zf.extractall(target)


def download_rcaeval(suite, files):
    suite_dir = RCAEVAL_DIR / suite
    suite_dir.mkdir(parents=True, exist_ok=True)
    for system in SYSTEMS:
        name = f"{suite}-{system}"
        marker = suite_dir / f"{name}{COMPLETE_MARKER}"
        if marker.exists():
            print(f"{name}: already downloaded")
            continue
        meta = files[f"{name}.zip"]
        archive = suite_dir / f"{name}.zip"
        print(f"{name}: downloading")
        fetch(meta["links"]["self"], archive, meta["checksum"].removeprefix("md5:"), meta["size"])
        safe_extract(archive, suite_dir)
        archive.unlink()
        marker.touch()


def check(label, ok, detail):
    print(f"  [{'ok' if ok else 'FAIL'}] {label}: {detail}")
    return ok


def verify_smd():
    print("SMD checks")
    ok = True
    for split in SMD_SPLITS:
        count = len(list((SMD_DIR / split).glob("machine-*.txt")))
        ok &= check(f"{split} machines", count == SMD_MACHINES, count)

    train = np.loadtxt(SMD_DIR / "train" / "machine-1-1.txt", delimiter=",")
    test = np.loadtxt(SMD_DIR / "test" / "machine-1-1.txt", delimiter=",")
    label = np.loadtxt(SMD_DIR / "test_label" / "machine-1-1.txt", delimiter=",")
    ok &= check("machine-1-1 train columns", train.shape[1] == SMD_METRICS, train.shape[1])
    ok &= check("machine-1-1 test columns", test.shape[1] == SMD_METRICS, test.shape[1])
    ok &= check("machine-1-1 test rows == label rows", len(test) == len(label), f"{len(test)} vs {len(label)}")
    print(f"  [info] machine-1-1 anomaly fraction: {label.mean():.4f}")

    labels = [np.loadtxt(p, delimiter=",") for p in sorted((SMD_DIR / "test_label").glob("machine-*.txt"))]
    overall = np.concatenate(labels).mean()
    ok &= check("anomaly fraction, all 28 test sets", abs(overall - 0.0416) < 0.001, f"{overall:.4f}")
    return ok


def case_dirs(suite):
    """Cases live at <suite>-<system>/<service>_<fault>/<instance number>/.

    The instance must be a number: the RE2-OB archive also ships a stray copy of
    RCAEval's multi-source sample (with its own inject_time.txt) inside
    checkoutservice_cpu/, and it is not a benchmark case.
    """
    return sorted(p.parent for p in (RCAEVAL_DIR / suite).glob("*/*/*/inject_time.txt")
                  if p.parent.name.isdigit())


def verify_rcaeval(suite):
    print(f"{suite} checks")
    cases = case_dirs(suite)
    ok = check("cases", len(cases) == RCAEVAL_CASES[suite], f"{len(cases)} (expected {RCAEVAL_CASES[suite]})")
    missing = [c for c in cases if not any(c.glob("*.csv"))]
    ok &= check("cases with telemetry", not missing, f"{len(cases) - len(missing)} of {len(cases)}")
    unreadable = [c for c in cases if not (c / "inject_time.txt").read_text().strip().isdigit()]
    ok &= check("inject times readable", not unreadable, f"{len(cases) - len(unreadable)} of {len(cases)}")
    return ok


def downloaded_suites():
    return [s for s in RCAEVAL_CASES
            if all((RCAEVAL_DIR / s / f"{s}-{system}{COMPLETE_MARKER}").exists() for system in SYSTEMS)]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rcaeval", nargs="*", default=["RE1"], choices=["RE1", "RE2", "RE3"],
                        help="RCAEval suites to download (default: RE1)")
    parser.add_argument("--skip-smd", action="store_true", help="do not download SMD")
    parser.add_argument("--verify-only", action="store_true", help="run checks without downloading")
    args = parser.parse_args()

    DATA.mkdir(exist_ok=True)
    if not args.verify_only:
        if not args.skip_smd:
            download_smd()
        if args.rcaeval:
            files = zenodo_files()
            for suite in args.rcaeval:
                download_rcaeval(suite, files)

    ok = verify_smd() if (SMD_DIR / COMPLETE_MARKER).exists() else True
    for suite in downloaded_suites():
        ok &= verify_rcaeval(suite)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
