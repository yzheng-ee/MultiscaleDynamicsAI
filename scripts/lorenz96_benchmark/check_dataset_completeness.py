"""Lightweight, read-only check of the complete local or transferred benchmark.

Run from the repository root using environments/lorenz96_benchmark.yml:
    python scripts/lorenz96_benchmark/check_dataset_completeness.py

Checks the expected inventory, JSON records and references, paired identities,
closure hashes, and NPZ array headers/shapes without loading trajectory arrays.
No files are written. Cache files and other unrelated files are ignored.
This is a completeness check, not a full payload checksum or numerical validation;
corruption inside compressed trajectory payloads requires a full integrity scan.
Exit codes: 0 = complete, 1 = errors found.
"""

import hashlib
import json
from math import prod
from pathlib import Path
import struct
from time import perf_counter
from zipfile import ZipFile

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "data/lorenz96_benchmark"
SETTINGS = ("a", "b_f", "b_hx", "b_hy", "b_j", "b_k")
# Directory, configuration, number of pairs, first initialization seed.
COLLECTIONS = (
    ("training", "a", 32, 1001),
    ("validation", "a", 8, 2001),
    ("test/test_a", "a", 16, 3001),
    ("test/test_b_k", "b_k", 4, 4001),
    ("test/test_b_j", "b_j", 4, 5001),
    ("test/test_b_hx", "b_hx", 4, 6001),
    ("test/test_b_hy", "b_hy", 4, 7001),
    ("test/test_b_f", "b_f", 4, 8001),
)
REFERENCE_KEYS = {"source_trajectory", "source_metadata", "source_closure", "bandwidth_source"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check_npz(path, shape):
    """Read only ZIP directory entries and the small NPY headers."""
    with ZipFile(path) as archive, path.open("rb") as raw:
        require(sorted(archive.namelist()) == ["state.npy", "time.npy"],
                "Expected exactly time.npy and state.npy")
        for name, expected_shape in (("time.npy", (shape[1],)), ("state.npy", tuple(shape))):
            entry = archive.getinfo(name)
            # Check that the compressed member fits before the ZIP directory.
            raw.seek(entry.header_offset)
            header = raw.read(30)
            require(len(header) == 30 and header[:4] == b"PK\x03\x04", "Invalid ZIP member header")
            name_length, extra_length = struct.unpack_from("<HH", header, 26)
            end = entry.header_offset + 30 + name_length + extra_length + entry.compress_size
            require(end <= archive.start_dir, "Truncated or overlapping ZIP member")
            with archive.open(entry) as stream:
                version = np.lib.format.read_magic(stream)
                if version == (1, 0):
                    actual_shape, _, dtype = np.lib.format.read_array_header_1_0(stream)
                elif version == (2, 0):
                    actual_shape, _, dtype = np.lib.format.read_array_header_2_0(stream)
                else:
                    raise ValueError(f"Unexpected NPY version: {version}")
                require(actual_shape == expected_shape, f"{name}: unexpected shape {actual_shape}")
                require(dtype.kind == "f" and dtype.itemsize == 8, f"{name}: expected float64")
                require(stream.tell() + prod(actual_shape) * dtype.itemsize == entry.file_size,
                        f"{name}: inconsistent uncompressed size")


class CompletenessCheck:
    def __init__(self):
        self.errors = []
        self.expected = set()
        self.npz_count = 0
        self.hashes = {}

    def run(self, path, operation):
        """Continue checking independent files after a failure."""
        try:
            return operation()
        except (OSError, ValueError, KeyError, TypeError, IndexError, EOFError,
                struct.error, RuntimeError) as error:
            self.errors.append(f"{path.relative_to(DATASET)}: {error}")
        except Exception as error:
            # BadZipFile and decompression errors should also appear in the summary.
            self.errors.append(f"{path.relative_to(DATASET)}: {type(error).__name__}: {error}")
        return None

    def file(self, path):
        self.expected.add(path)
        require(path.is_file() and path.stat().st_size > 0, "Missing or empty file")

    def references(self, value, directory):
        if isinstance(value, dict):
            for key, item in value.items():
                if key in REFERENCE_KEYS and isinstance(item, str):
                    target = ROOT / item if item.startswith("data/") else directory / item
                    require(target.is_file() and target.stat().st_size > 0,
                            f"Missing or empty reference: {item}")
                self.references(item, directory)
            if "source_closure" in value and "sha256" in value:
                target = ROOT / value["source_closure"]
                if target not in self.hashes:
                    self.hashes[target] = hashlib.sha256(target.read_bytes()).hexdigest()
                require(self.hashes[target] == value["sha256"], "Closure SHA-256 mismatch")
        elif isinstance(value, list):
            for item in value:
                self.references(item, directory)

    def json(self, path, check_references=True):
        self.file(path)
        value = json.loads(path.read_text())
        if check_references:
            self.references(value, path.parent)
        return value

    def trajectory(self, directory, record, setting, family, samples):
        require(record["config_id"] == f"config_{setting}", "Configuration mismatch")
        require(record["family"] == ("single_scale" if family == "singlescale" else family),
                "Family mismatch")
        parameters = record["parameters"]
        require(parameters["K"] == (18 if setting == "b_k" else 9)
                and parameters["J"] == (16 if setting == "b_j" else 8),
                "Unexpected channel configuration")
        channels = parameters["K"] * (1 + parameters["J"]) if family == "multiscale" else parameters["K"]
        shape = [channels, samples]
        require(record["state_shape"] == shape, "Metadata shape mismatch")
        filename = record["filename"]
        require(Path(filename).name == filename and filename.endswith(".npz"), "Invalid trajectory filename")
        path = directory / filename
        self.file(path)
        check_npz(path, shape)
        self.npz_count += 1

    def calibration(self, family, setting):
        directory = DATASET / "dense_calibration" / family / f"{family}_dense_calibration_trajectory_{setting}"
        record = self.json(directory / "metadata.json")
        self.run(directory / record["filename"],
                 lambda: self.trajectory(directory, record, setting, family, 1_000_000))
        bandwidth = self.json(directory / "bandwidth_periods.json")
        require(bandwidth["config_id"] == record["config_id"], "Bandwidth configuration mismatch")
        require(len(bandwidth["channel_bandwidth_periods"]) == record["state_shape"][0],
                "Incomplete channel bandwidth records")

    def collection(self, relative, setting, count, first_seed):
        records_by_family = {}
        for family in ("multiscale", "singlescale", "slow_downsampled"):
            directory = DATASET / relative / (family if family != "slow_downsampled" else "multiscale/slow_downsampled")
            records = self.run(directory / "metadata.json", lambda: self.json(directory / "metadata.json"))
            if records is None:
                continue
            def check_records():
                require(isinstance(records, list) and len(records) == count, f"Expected {count} records")
                for key in ("filename", "trajectory_id", "pair_id"):
                    require(len({r[key] for r in records}) == count, f"Duplicate {key}")
            if self.run(directory / "metadata.json", lambda: (check_records(), True)[1]) is None:
                continue
            records_by_family[family] = {r["pair_id"]: r for r in records}
            for record in records:
                def check_record():
                    if family == "multiscale":
                        samples = 8191 * record["sampling"]["fast_intervals_per_slow_interval"] + 1
                    else:
                        samples = 8192
                    self.trajectory(directory, record, setting,
                                    "singlescale" if family == "slow_downsampled" else family, samples)
                # Derived views retain family=multiscale but contain only K channels.
                if family == "slow_downsampled":
                    def check_record():
                        require(record["view"] == "slow_downsampled", "Incorrect derived view")
                        require(record["family"] == "multiscale", "Incorrect derived family")
                        copy = dict(record, family="single_scale")
                        self.trajectory(directory, copy, setting, "singlescale", 8192)
                self.run(directory / record["filename"], check_record)
        def check_pairs():
            require(len(records_by_family) == 3, "Incomplete trajectory families or derived views")
            multi, single, slow = (records_by_family[f] for f in ("multiscale", "singlescale", "slow_downsampled"))
            require(multi.keys() == single.keys() == slow.keys(), "Pair IDs differ")
            require({r["settings"]["seed"] for r in multi.values()} == set(range(first_seed, first_seed + count)),
                    "Unexpected initialization seeds")
            for pair in multi:
                m, s, d = multi[pair], single[pair], slow[pair]
                require(m["parameters"] == s["parameters"] == d["parameters"], "Paired parameters differ")
                require(m["settings"]["seed"] == s["settings"]["seed"], "Paired seeds differ")
                require(m["initialization"] == s["initialization"], "Paired initialization differs")
                require(m["paired_singlescale_trajectory"]["trajectory_id"] == s["trajectory_id"]
                        and s["paired_multiscale_trajectory"]["trajectory_id"] == m["trajectory_id"],
                        "Incorrect partner identity")
                require(d["filename"] == m["filename"] and d["time_range"] == m["time_range"],
                        "Derived view differs from source metadata")
        self.run(DATASET / relative, check_pairs)


def main():
    started = perf_counter()
    check = CompletenessCheck()
    check.run(DATASET / "README.md", lambda: check.file(DATASET / "README.md"))
    # The historical report embeds bandwidth records with calibration-local paths.
    # Their current originals are checked in their own directories below.
    report = check.run(DATASET / "validation_report.json",
                       lambda: check.json(DATASET / "validation_report.json", check_references=False))
    if report is not None:
        check.run(DATASET / "validation_report.json", lambda: require(
            report["status"] == "pass" and report["scope"]["full_dataset_validation"],
            "Stored numerical validation did not pass the full dataset"))
    for setting in SETTINGS:
        directory = DATASET / "closure" / setting
        def check_closure():
            record = check.json(directory / "metadata.json")
            require(record["config_id"] == f"config_{setting}" and record["sampling"]["sample_size"] == 800,
                    "Incorrect closure configuration or training sample count")
            check.file(directory / "closure.joblib")
        check.run(directory, check_closure)
        for family in ("multiscale", "singlescale"):
            check.run(DATASET / "dense_calibration" / family / f"{family}_dense_calibration_trajectory_{setting}",
                      lambda: check.calibration(family, setting))
    for relative, setting, count, first_seed in COLLECTIONS:
        check.run(DATASET / relative, lambda: check.collection(relative, setting, count, first_seed))
    # Detect unlisted data artifacts, while ignoring caches and transfer leftovers.
    actual = {p for p in DATASET.rglob("*") if p.is_file() and p.suffix in {".npz", ".json", ".joblib"}}
    for path in sorted(actual - check.expected):
        check.errors.append(f"{path.relative_to(DATASET)}: Unexpected dataset artifact")
    for error in check.errors:
        print(f"ERROR: {error}")
    status = "FAIL" if check.errors else "PASS"
    print(f"{status}: {check.npz_count}/240 NPZ files checked; "
          f"{len(check.hashes)}/6 closure hashes checked; {len(check.errors)} errors; "
          f"{perf_counter() - started:.2f}s elapsed. No files written.")
    return int(bool(check.errors))


if __name__ == "__main__":
    raise SystemExit(main())
