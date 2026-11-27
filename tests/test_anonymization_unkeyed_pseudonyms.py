"""Security spec for dataset pseudonyms.

A recipient of the released parquet must not be able to recover identifiers
from that file alone. These tests fail while pseudonyms are an unkeyed hash
of the raw identifier.
"""

import ast
import hashlib
import hmac
import os
import re
import shutil
import tempfile
import time
import unittest
from pathlib import Path

import pandas as pd

from utils import anonyme_dataset_coro
from utils.anonyme_dataset_coro import anonymize_value, anonymize_with_tracking, main

_HEX_RUN = re.compile(r"[0-9a-f]{16,}")
_DIGEST_ALGOS = ("md5", "sha1", "sha256", "sha384", "sha512", "blake2s", "blake2b")
_TRUNCATIONS = (16, 32)

# Every column the anonymizer hashes. Parseable PatientBirthDate values take
# the year-only branch and are not part of this pseudonym construction.
_HASHED_COLUMNS = {
    "StudyInstanceUID": "1.2.840.999999.100",
    "SeriesInstanceUID": "1.2.840.999999.100.1",
    "SOPInstanceUID": "1.2.840.999999.100.1.1",
    "CathReport_MRN": "4829107",
    "Patient_ID": "7733119",
    "IDPATIENT": "0048291",
    "PatientName": "SYNTHETIC_PATIENT_ALPHA",
    "PatientBirthDate": "not-a-date",
    "PatientBirthTime": "101500",
    "InstitutionName": "SYNTHETIC_GENERAL_HOSPITAL",
    "InstitutionAddress": "1 Synthetic Way",
    "StationName": "SYNTH_STATION_A",
    "DeviceSerialNumber": "SN-10001",
    "HOSPITAL": "SYNTHETIC_COMMUNITY_HOSPITAL",
    "SALLE": "SALLE-A",
    "NSEJOUR": "J100001",
    "DICOMPath": "/synthetic/path/a",
    "FileName": "synthetic_a.dcm",
    "StudyID": "STUDY42",
    "NameOfPhysiciansReadingStudy": "DR_SYNTHETIC_READER",
    "PerformingPhysicianName": "DR_SYNTHETIC_OPERATOR",
    "OperatorsName": "SYNTHETIC_OPERATOR",
}

_NUMERIC_COLUMNS = ("CathReport_MRN", "Patient_ID", "IDPATIENT")


def _module_literals(path: Path) -> list[bytes]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    literals: list[bytes] = []
    seen: set[bytes] = set()
    for node in ast.walk(tree):
        value = node.value if isinstance(node, ast.Constant) else None
        if isinstance(value, str):
            encoded = value.encode("utf-8")
        elif isinstance(value, bytes):
            encoded = value
        else:
            continue
        if 0 < len(encoded) <= 512 and encoded not in seen:
            seen.add(encoded)
            literals.append(encoded)
    return literals


def _public_blobs(value: object, column: str, literals: list[bytes]) -> list[bytes]:
    raw = str(value).encode()
    keys = [
        b"",
        column.encode(),
        column.upper().encode(),
        column.split("_")[0].encode(),
        *literals,
    ]
    blobs: list[bytes] = []
    seen: set[bytes] = set()
    for key in keys:
        for blob in (
            raw,
            key + raw,
            raw + key,
            key + b":" + raw,
            key + b"|" + raw,
            key + b"_" + raw,
        ):
            if blob and blob not in seen:
                seen.add(blob)
                blobs.append(blob)
    return blobs


def matches_public_pseudonym(
    token: object, value: object, column: str, literals: list[bytes]
) -> bool:
    """True when a recipient can recompute the token from the identifier alone."""
    released = str(token).lower()
    raw = str(value).encode()
    if str(value) != "" and str(value) in str(token):
        return True
    for blob in _public_blobs(value, column, literals):
        for algo in _DIGEST_ALGOS:
            digest = hashlib.new(algo, blob).hexdigest()
            for size in _TRUNCATIONS:
                if size <= len(digest) and digest[:size] in released:
                    return True
    for key in literals:
        for algo in ("md5", "sha256"):
            digest = hmac.new(key, raw, algo).hexdigest()
            if digest[:16] in released or digest[:32] in released:
                return True
    return False


def recover_numeric_by_unkeyed_md5(tokens: list[str], digits: int = 7) -> dict[str, str]:
    """Brute-force short numeric identifiers from released tokens.

    The search uses only the released strings. It does not receive the originals.
    """
    by_digest: dict[str, list[str]] = {}
    for token in tokens:
        for run in _HEX_RUN.findall(token.lower()):
            by_digest.setdefault(run[:16], []).append(token)
    if not by_digest:
        return {}

    found: dict[str, str] = {}
    remaining = set(by_digest)
    for i in range(10**digits):
        plain = str(i)
        padded = f"{i:0{digits}d}"
        candidates = [(hashlib.md5(plain.encode()).hexdigest()[:16], plain)]
        if padded != plain:
            candidates.append((hashlib.md5(padded.encode()).hexdigest()[:16], padded))
        for digest, text in candidates:
            if digest not in remaining:
                continue
            for token in by_digest[digest]:
                found[token] = text
            remaining.discard(digest)
        if not remaining:
            break
    return found


def _frame_from_identifiers(identifiers: dict[str, str], clinical: str) -> pd.DataFrame:
    row = dict(identifiers)
    row["stenosis_percent"] = clinical
    return pd.DataFrame([row])


class TestReleasedParquetPseudonyms(unittest.TestCase):
    """Run the script entry point and attack only the parquet it writes."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.literals = _module_literals(Path(anonyme_dataset_coro.__file__))
        cls.temp_dir = tempfile.mkdtemp()
        cls._previous_cwd = os.getcwd()
        os.chdir(cls.temp_dir)
        try:
            _write_product_inputs(Path(cls.temp_dir))
            main()
            released_path = Path("reports/anonymized_post_pci_dataset_2017-2024.parquet")
            linkage_path = Path("reports/anonymization_mapping_post_pci.json")
            cls.released = pd.read_parquet(released_path)
            cls.ground_truth = pd.read_csv("reports/processed_dataset.csv")
            linkage_path.unlink()
            cls.released = pd.read_parquet(released_path)
        except Exception:
            os.chdir(cls._previous_cwd)
            raise

    @classmethod
    def tearDownClass(cls) -> None:
        os.chdir(cls._previous_cwd)
        shutil.rmtree(cls.temp_dir, ignore_errors=True)

    def test_numeric_identifiers_are_not_recovered_from_released_parquet(self) -> None:
        tokens: list[str] = []
        for column in _NUMERIC_COLUMNS:
            tokens.extend(str(value) for value in self.released[column].tolist())
        started = time.perf_counter()
        found = recover_numeric_by_unkeyed_md5(tokens, digits=7)
        elapsed = time.perf_counter() - started

        linked = []
        for index in range(len(self.released)):
            prediction = self.released.loc[index, "stenosis_pred"]
            for column in _NUMERIC_COLUMNS:
                token = str(self.released.loc[index, column])
                if token in found:
                    linked.append((column, found[token], prediction))
        self.assertEqual(
            linked,
            [],
            f"recovered identifiers linked to predictions in {elapsed:.2f}s "
            "from the released parquet after the linkage file was removed",
        )

    def test_released_columns_are_not_public_pseudonyms(self) -> None:
        for column in _HASHED_COLUMNS:
            for index in range(len(self.released)):
                with self.subTest(column=column, row=index):
                    original = self.ground_truth.loc[index, column]
                    token = self.released.loc[index, column]
                    self.assertFalse(
                        matches_public_pseudonym(token, original, column, self.literals)
                    )


class TestAnonymizeWithTracking(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.literals = _module_literals(Path(anonyme_dataset_coro.__file__))
        cls.temp_dir = tempfile.mkdtemp()
        frame = _frame_from_identifiers(_HASHED_COLUMNS, clinical="70")
        mapping_path = Path(cls.temp_dir) / "linkage.json"
        cls.anonymized, _log = anonymize_with_tracking(frame, mapping_filepath=str(mapping_path))
        mapping_path.unlink()
        cls.original = frame

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.temp_dir, ignore_errors=True)

    def test_tracking_output_is_not_a_public_pseudonym(self) -> None:
        for column in _HASHED_COLUMNS:
            with self.subTest(column=column):
                self.assertFalse(
                    matches_public_pseudonym(
                        self.anonymized.loc[0, column],
                        self.original.loc[0, column],
                        column,
                        self.literals,
                    )
                )

    def test_string_identifier_with_leading_zeros_is_not_enumerable(self) -> None:
        token = str(self.anonymized.loc[0, "IDPATIENT"])
        found = recover_numeric_by_unkeyed_md5([token], digits=7)
        self.assertNotIn(token, found)


class TestAnonymizeValueFallback(unittest.TestCase):
    """anonymize_value hashes on its own when the linkage map has no entry."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.literals = _module_literals(Path(anonyme_dataset_coro.__file__))

    def test_fallback_is_not_a_public_pseudonym(self) -> None:
        for column, value in _HASHED_COLUMNS.items():
            with self.subTest(column=column):
                token = anonymize_value(value, column, {})
                self.assertFalse(matches_public_pseudonym(token, value, column, self.literals))

    def test_integer_mrn_fallback_is_not_enumerable(self) -> None:
        token = str(anonymize_value(4829107, "CathReport_MRN", {}))
        found = recover_numeric_by_unkeyed_md5([token], digits=7)
        self.assertNotIn(token, found)


def _write_product_inputs(root: Path) -> None:
    reports = root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    ground_truth = pd.DataFrame(
        [
            _ground_truth_row(
                file_name="synthetic_a.dcm",
                study_uid="1.2.840.999999.100",
                series_uid="1.2.840.999999.100.1",
                sop_uid="1.2.840.999999.100.1.1",
                mrn=4829107,
                patient_id=7733119,
                id_patient=1592634,
                patient_name="SYNTHETIC_PATIENT_ALPHA",
                hospital="SYNTHETIC_GENERAL_HOSPITAL",
                site="SYNTHETIC_COMMUNITY_HOSPITAL",
                stenosis=70,
            ),
            _ground_truth_row(
                file_name="synthetic_b.dcm",
                study_uid="1.2.840.999999.200",
                series_uid="1.2.840.999999.200.1",
                sop_uid="1.2.840.999999.200.1.1",
                mrn=2581470,
                patient_id=8642097,
                id_patient=3692581,
                patient_name="SYNTHETIC_PATIENT_BETA",
                hospital="SYNTHETIC_REGIONAL_HOSPITAL",
                site="SYNTHETIC_DAY_HOSPITAL",
                stenosis=40,
            ),
        ]
    )
    ground_truth.to_csv(reports / "processed_dataset.csv", index=False)
    inference = pd.DataFrame(
        {
            "video_name": ["['synthetic_a.dcm']", "['synthetic_b.dcm']"],
            "stenosis_pred": [70, 40],
        }
    )
    inference.to_csv(root / "post_pci_inference_2017-2024.csv", index=False)


def _ground_truth_row(
    file_name: str,
    study_uid: str,
    series_uid: str,
    sop_uid: str,
    mrn: int,
    patient_id: int,
    id_patient: int,
    patient_name: str,
    hospital: str,
    site: str,
    stenosis: int,
) -> dict[str, object]:
    return {
        "FileName": file_name,
        "StudyInstanceUID": study_uid,
        "SeriesInstanceUID": series_uid,
        "SOPInstanceUID": sop_uid,
        "CathReport_MRN": mrn,
        "Patient_ID": patient_id,
        "IDPATIENT": id_patient,
        "PatientName": patient_name,
        "PatientBirthDate": "not-a-date",
        "PatientBirthTime": "101500",
        "InstitutionName": hospital,
        "InstitutionAddress": "1 Synthetic Way",
        "StationName": "SYNTH_STATION_A",
        "DeviceSerialNumber": "SN-10001",
        "HOSPITAL": site,
        "SALLE": "SALLE-A",
        "NSEJOUR": "J100001",
        "DICOMPath": f"/synthetic/path/{file_name}",
        "StudyID": "STUDY42",
        "NameOfPhysiciansReadingStudy": "DR_SYNTHETIC_READER",
        "PerformingPhysicianName": "DR_SYNTHETIC_OPERATOR",
        "OperatorsName": "SYNTHETIC_OPERATOR",
        "stenosis_percent": stenosis,
    }
