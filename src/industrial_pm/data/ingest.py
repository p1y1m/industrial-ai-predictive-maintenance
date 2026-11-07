from __future__ import annotations

import hashlib
import json
import shutil
import urllib.request
import zipfile
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from industrial_pm.data.constants import (
    NASA_CMAPSS_URL,
    NASA_DATASET_CITATION,
    SUPPORTED_SUBSETS,
)
from industrial_pm.data.schema import (
    add_test_rul,
    add_train_rul,
    read_cmapss_table,
    validate_cmapss_frame,
)


@dataclass(frozen=True)
class DatasetManifest:
    dataset: str
    subset: str
    source_type: str
    source_url: str
    citation: str
    retrieved_at: str
    archive_sha256: str
    files_sha256: dict[str, str]
    train_rows: int
    train_engines: int
    test_rows: int
    test_engines: int
    license_note: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_archive(destination: Path, url: str = NASA_CMAPSS_URL) -> Path:
    """Download the official archive atomically; reuse an existing nonempty cache."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0:
        return destination
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "industrial-pm/0.1"})
    with urllib.request.urlopen(request, timeout=60) as response, temporary.open("wb") as output:
        shutil.copyfileobj(response, output)
    if temporary.stat().st_size < 1_000_000:
        raise RuntimeError("Downloaded archive is unexpectedly small")
    temporary.replace(destination)
    return destination


def safe_extract_archive(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            target = (destination / member.filename).resolve()
            if root not in target.parents and target != root:
                raise ValueError(f"Unsafe archive member: {member.filename}")
        bundle.extractall(destination)


def prepare_official_subset(
    data_root: Path, subset: str = "FD001", rul_cap: int = 125
) -> DatasetManifest:
    subset = subset.upper()
    if subset not in SUPPORTED_SUBSETS:
        raise ValueError(f"subset must be one of {SUPPORTED_SUBSETS}")

    raw_dir = data_root / "raw"
    interim_dir = data_root / "interim" / "cmapss"
    processed_dir = data_root / "processed" / subset.lower()
    archive = download_archive(raw_dir / "CMAPSSData.zip")
    expected_names = [f"train_{subset}.txt", f"test_{subset}.txt", f"RUL_{subset}.txt"]
    if not all(list(interim_dir.rglob(name)) for name in expected_names):
        safe_extract_archive(archive, interim_dir)
        # NASA currently wraps the standard archive inside a dataset-directory ZIP.
        for nested_archive in interim_dir.rglob("CMAPSSData.zip"):
            safe_extract_archive(nested_archive, interim_dir / "standard")

    required: list[Path] = []
    for name in expected_names:
        matches = list(interim_dir.rglob(name))
        if len(matches) != 1:
            raise FileNotFoundError(f"Expected exactly one {name}, found {len(matches)}")
        required.append(matches[0])

    train = read_cmapss_table(required[0])
    test = read_cmapss_table(required[1])
    train_quality = validate_cmapss_frame(train)
    test_quality = validate_cmapss_frame(test)
    train_labeled = add_train_rul(train, cap=rul_cap)
    test_labeled = add_test_rul(test, required[2], cap=rul_cap)

    processed_dir.mkdir(parents=True, exist_ok=True)
    train_path = processed_dir / "train.csv.gz"
    test_path = processed_dir / "test.csv.gz"
    train_labeled.to_csv(train_path, index=False, compression="gzip")
    test_labeled.to_csv(test_path, index=False, compression="gzip")

    file_hashes = {path.name: sha256_file(path) for path in required}
    file_hashes[train_path.name] = sha256_file(train_path)
    file_hashes[test_path.name] = sha256_file(test_path)
    manifest = DatasetManifest(
        dataset="NASA C-MAPSS Turbofan Engine Degradation Simulation",
        subset=subset,
        source_type="official_nasa_pcoe",
        source_url=NASA_CMAPSS_URL,
        citation=NASA_DATASET_CITATION,
        retrieved_at=datetime.now(UTC).isoformat(),
        archive_sha256=sha256_file(archive),
        files_sha256=file_hashes,
        train_rows=train_quality.rows,
        train_engines=train_quality.engines,
        test_rows=test_quality.rows,
        test_engines=test_quality.engines,
        license_note=(
            "NASA catalog access is public, but the C-MAPSS dataset record does not specify "
            "an explicit license. Raw data is not redistributed by this project."
        ),
    )
    (processed_dir / "manifest.json").write_text(
        json.dumps(asdict(manifest), indent=2) + "\n", encoding="utf-8"
    )
    quality = {"train": train_quality.to_dict(), "test": test_quality.to_dict()}
    (processed_dir / "quality_report.json").write_text(
        json.dumps(quality, indent=2) + "\n", encoding="utf-8"
    )
    return manifest
