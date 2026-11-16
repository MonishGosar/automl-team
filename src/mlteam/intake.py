"""Data intake: turn "a CSV, a folder of CSVs, a zip, or a Kaggle dataset" into
one read-only, hashed landing zone.

The rest of the pipeline never reads the user's original files. It reads the
copies under ``<run>/raw`` so a run can be reproduced exactly: the dataset hash
changes if and only if a file's bytes or name change.
"""

from __future__ import annotations

import glob
import hashlib
import os
import re
import shutil
import stat
import tempfile
import zipfile
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path, PurePosixPath

from mlteam.runstore import RunPaths, now_iso, write_json

SUPPORTED_SUFFIXES = (".csv", ".tsv", ".csv.gz", ".tsv.gz", ".parquet")
MAX_FILES = 200
MAX_ZIP_BYTES = 5 * 1024**3

KaggleFetcher = Callable[[str, str], Path]


class IntakeError(Exception):
    """Raised when the given sources cannot be turned into a dataset."""


@dataclass(frozen=True)
class RawFile:
    name: str
    sha256: str
    size_bytes: int
    origin: str


@dataclass
class Manifest:
    run_id: str
    created_at: str
    sources: list[str]
    files: list[RawFile]
    skipped: list[str] = field(default_factory=list)
    dataset_hash: str = ""

    def to_json(self) -> dict[str, object]:
        return asdict(self)


def is_supported(name: str) -> bool:
    return name.lower().endswith(SUPPORTED_SUFFIXES)


_KAGGLE_URL = re.compile(
    r"^https?://(?:www\.)?kaggle\.com/(datasets|competitions|c)/([^?#]+)", re.IGNORECASE
)


def parse_kaggle(spec: str) -> tuple[str, str] | None:
    """Recognise a Kaggle reference. Returns (kind, handle) or None.

    Accepted forms:
      kaggle:owner/dataset
      kaggle:competition/name   or   kaggle-competition:name
      https://www.kaggle.com/datasets/owner/dataset
      https://www.kaggle.com/competitions/name
    """
    text = spec.strip()
    lowered = text.lower()
    if lowered.startswith("kaggle-competition:"):
        return ("competition", text.split(":", 1)[1].strip("/ "))
    if lowered.startswith("kaggle:"):
        handle = text.split(":", 1)[1].strip("/ ")
        if handle.lower().startswith("competition/"):
            return ("competition", handle.split("/", 1)[1])
        if handle.count("/") != 1:
            raise IntakeError(
                f"Kaggle dataset must look like kaggle:owner/dataset, got: {spec}"
            )
        return ("dataset", handle)
    match = _KAGGLE_URL.match(text)
    if match:
        section, rest = match.group(1).lower(), match.group(2).strip("/")
        parts = rest.split("/")
        if section == "datasets":
            if len(parts) < 2:
                raise IntakeError(f"Kaggle dataset URL is missing owner/dataset: {spec}")
            return ("dataset", f"{parts[0]}/{parts[1]}")
        return ("competition", parts[0])
    return None


def fetch_from_kaggle(kind: str, handle: str) -> Path:
    """Download through kagglehub. Credentials stay in this process: they are
    read by kagglehub from the environment or ~/.kaggle, never handed to an agent."""
    try:
        import kagglehub
    except ImportError as exc:  # pragma: no cover - dependency is declared
        raise IntakeError("kagglehub is not installed") from exc
    try:
        if kind == "competition":
            return Path(kagglehub.competition_download(handle))
        return Path(kagglehub.dataset_download(handle))
    except Exception as exc:  # noqa: BLE001 - surface any download failure plainly
        raise IntakeError(
            f"Kaggle download failed for {handle}: {exc}\n"
            "If this dataset needs a login, set KAGGLE_API_TOKEN or run "
            "`python -c \"import kagglehub; kagglehub.login()\"`. "
            "Competition data also needs the rules accepted on kaggle.com."
        ) from exc


def _extract_zip(zip_path: Path, dest: Path, skipped: list[str]) -> list[Path]:
    dest.mkdir(parents=True, exist_ok=True)
    extracted: list[Path] = []
    total = 0
    try:
        archive = zipfile.ZipFile(zip_path)
    except zipfile.BadZipFile as exc:
        raise IntakeError(f"Not a valid zip file: {zip_path}") from exc
    with archive:
        for info in archive.infolist():
            if info.is_dir():
                continue
            member = PurePosixPath(info.filename.replace("\\", "/"))
            label = f"{zip_path.name}:{info.filename}"
            if member.is_absolute() or ".." in member.parts:
                skipped.append(f"{label} (unsafe path)")
                continue
            if not is_supported(member.name):
                skipped.append(f"{label} (unsupported type)")
                continue
            total += info.file_size
            if total > MAX_ZIP_BYTES:
                raise IntakeError(
                    f"{zip_path.name} expands to more than {MAX_ZIP_BYTES // 1024**3} GB"
                )
            target = dest / "__".join(member.parts)
            with archive.open(info) as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            extracted.append(target)
    return extracted


def _collect_dir(folder: Path, scratch: Path, skipped: list[str]) -> list[Path]:
    found: list[Path] = []
    for path in sorted(p for p in folder.rglob("*") if p.is_file()):
        if is_supported(path.name):
            found.append(path)
        elif path.suffix.lower() == ".zip":
            found.extend(_extract_zip(path, scratch / f"zip_{len(os.listdir(scratch))}", skipped))
        else:
            skipped.append(f"{path.name} (unsupported type)")
    return found


def collect(
    spec: str,
    scratch: Path,
    skipped: list[str],
    fetch_kaggle: KaggleFetcher,
) -> list[tuple[Path, str]]:
    """Resolve one source spec into concrete data files, each tagged with its origin."""
    kaggle = parse_kaggle(spec)
    if kaggle is not None:
        kind, handle = kaggle
        folder = fetch_kaggle(kind, handle)
        origin = f"kaggle:{kind}:{handle}"
        if folder.is_file():
            return [(folder, origin)] if is_supported(folder.name) else []
        return [(p, origin) for p in _collect_dir(folder, scratch, skipped)]

    path = Path(spec).expanduser()
    if path.is_dir():
        return [(p, str(path)) for p in _collect_dir(path, scratch, skipped)]
    if path.is_file():
        if path.suffix.lower() == ".zip":
            files = _extract_zip(path, scratch / f"zip_{len(os.listdir(scratch))}", skipped)
            return [(p, str(path)) for p in files]
        if is_supported(path.name):
            return [(path, str(path))]
        raise IntakeError(
            f"Unsupported file type: {path.name}. Supported: {', '.join(SUPPORTED_SUFFIXES)}, .zip"
        )
    if any(ch in spec for ch in "*?["):
        matches = sorted(Path(m) for m in glob.glob(spec, recursive=True))
        files = [(m, spec) for m in matches if m.is_file() and is_supported(m.name)]
        if files:
            return files
        raise IntakeError(f"Pattern matched no data files: {spec}")
    raise IntakeError(
        f"Not found: {spec}\n"
        "Give a file, a folder, a glob like data/*.csv, a .zip, "
        "or a Kaggle dataset as kaggle:owner/dataset."
    )


def _unique_name(path: Path, used: set[str]) -> str:
    name = path.name
    if name.lower() not in used:
        return name
    parent = re.sub(r"[^A-Za-z0-9_-]+", "_", path.parent.name) or "dup"
    candidate = f"{parent}__{name}"
    counter = 2
    while candidate.lower() in used:
        candidate = f"{parent}_{counter}__{name}"
        counter += 1
    return candidate


def _copy_with_hash(src: Path, dest: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with src.open("rb") as reader, dest.open("wb") as writer:
        while chunk := reader.read(1024 * 1024):
            digest.update(chunk)
            writer.write(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def ingest(
    sources: Sequence[str],
    run: RunPaths,
    *,
    fetch_kaggle: KaggleFetcher = fetch_from_kaggle,
) -> Manifest:
    """Copy every data file behind ``sources`` into ``run.raw``, read-only and hashed."""
    if not sources:
        raise IntakeError("No data source given.")
    run.raw.mkdir(parents=True, exist_ok=True)
    skipped: list[str] = []
    files: list[RawFile] = []
    with tempfile.TemporaryDirectory(prefix="mlteam_intake_") as tmp:
        scratch = Path(tmp)
        found: list[tuple[Path, str]] = []
        seen: set[Path] = set()
        for spec in sources:
            for path, origin in collect(spec, scratch, skipped, fetch_kaggle):
                resolved = path.resolve()
                if resolved in seen:
                    continue
                seen.add(resolved)
                found.append((path, origin))
        if not found:
            raise IntakeError(
                "No CSV, TSV or Parquet files found in: " + ", ".join(sources)
            )
        if len(found) > MAX_FILES:
            raise IntakeError(
                f"Found {len(found)} data files; the limit is {MAX_FILES}. Point at a smaller folder."
            )
        used: set[str] = set()
        for path, origin in found:
            name = _unique_name(path, used)
            used.add(name.lower())
            dest = run.raw / name
            sha, size = _copy_with_hash(path, dest)
            os.chmod(dest, stat.S_IREAD)
            files.append(RawFile(name=name, sha256=sha, size_bytes=size, origin=origin))

    files.sort(key=lambda f: f.name.lower())
    digest = hashlib.sha256()
    for item in files:
        digest.update(f"{item.name}:{item.sha256}\n".encode())
    manifest = Manifest(
        run_id=run.run_id,
        created_at=now_iso(),
        sources=list(sources),
        files=files,
        skipped=skipped,
        dataset_hash=digest.hexdigest(),
    )
    write_json(run.manifest, manifest.to_json())
    return manifest
