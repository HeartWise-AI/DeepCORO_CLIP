"""Saved notebook outputs must not render cohort identifiers.

Readers of the repository get these outputs without running the notebooks.
The tests describe the safe state and fail while those outputs are still committed.
"""

import ast
import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

IDENTIFIER_COLUMNS = ("CathReport_MRN", "EXAMEN_ID", "patient_id_anon")
DICOM_FILE = re.compile(r"\.dcm(?:\.avi)?", re.IGNORECASE)
DICOM_UID = re.compile(r"\d+(?:\.\d+){6,}")
MEDIA_PATH = re.compile(r"/media/\S+")
STENOSIS = re.compile(r"stenosis", re.IGNORECASE)

FRAME_NOTEBOOKS = (
    "manuscript/analyses/explore_predictions.ipynb",
    "manuscript/analyses/check_multihead.ipynb",
)
REPORT_NOTEBOOK = "dev_scripts/generate_dataset.ipynb"
STUDY_PATH_SCRIPT = "manuscript/analyses/plot_middle_frame.py"
CI_WORKFLOW = ".github/workflows/tests.yml"


def _notebook_paths() -> list[Path]:
    return sorted(REPO_ROOT.rglob("*.ipynb"))


def _as_text(value: object) -> str:
    if isinstance(value, list):
        return "".join(part for part in value if isinstance(part, str))
    if isinstance(value, str):
        return value
    return ""


def _output_texts(notebook: dict) -> list[tuple[int, int, str, str]]:
    """Return (cell, output, mime, text) for every saved output string."""
    found: list[tuple[int, int, str, str]] = []
    for cell_index, cell in enumerate(notebook.get("cells", [])):
        for output_index, output in enumerate(cell.get("outputs") or []):
            output_type = output.get("output_type")
            if output_type == "stream":
                found.append((cell_index, output_index, "stream", _as_text(output.get("text"))))
            elif output_type == "error":
                traceback = _as_text(output.get("traceback"))
                found.append((cell_index, output_index, "error", traceback))
            elif output_type in ("execute_result", "display_data"):
                data = output.get("data") or {}
                for mime, payload in data.items():
                    if mime == "image/png":
                        continue
                    found.append((cell_index, output_index, mime, _as_text(payload)))
    return found


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _where(path: Path, cell_index: int, output_index: int, mime: str) -> str:
    relative = path.relative_to(REPO_ROOT)
    return f"{relative} cell {cell_index} output {output_index} ({mime})"


def test_notebook_outputs_do_not_render_identifier_columns() -> None:
    hits: list[str] = []
    for path in _notebook_paths():
        for cell_index, output_index, mime, text in _output_texts(_load(path)):
            present = [column for column in IDENTIFIER_COLUMNS if column in text]
            if present:
                hits.append(f"{_where(path, cell_index, output_index, mime)} columns={present}")
    assert hits == []


def test_notebook_outputs_do_not_render_dicom_study_paths() -> None:
    hits: list[str] = []
    for path in _notebook_paths():
        for cell_index, output_index, mime, text in _output_texts(_load(path)):
            if DICOM_FILE.search(text) and DICOM_UID.search(text):
                hits.append(_where(path, cell_index, output_index, mime))
    assert hits == []


def test_notebook_outputs_do_not_render_media_paths() -> None:
    hits: list[str] = []
    for path in _notebook_paths():
        for cell_index, output_index, mime, text in _output_texts(_load(path)):
            if MEDIA_PATH.search(text):
                hits.append(_where(path, cell_index, output_index, mime))
    assert hits == []


def test_generate_dataset_outputs_do_not_print_finding_reports() -> None:
    path = REPO_ROOT / REPORT_NOTEBOOK
    hits: list[str] = []
    for cell_index, output_index, mime, text in _output_texts(_load(path)):
        if STENOSIS.search(text):
            hits.append(_where(path, cell_index, output_index, mime))
    assert hits == []


def test_frame_notebooks_do_not_store_rendered_frames() -> None:
    hits: list[str] = []
    for relative in FRAME_NOTEBOOKS:
        path = REPO_ROOT / relative
        notebook = _load(path)
        for cell_index, cell in enumerate(notebook.get("cells", [])):
            for output_index, output in enumerate(cell.get("outputs") or []):
                data = output.get("data") or {}
                if "image/png" in data:
                    hits.append(_where(path, cell_index, output_index, "image/png"))
    assert hits == []


def test_plot_middle_frame_has_no_hardcoded_study_file() -> None:
    path = REPO_ROOT / STUDY_PATH_SCRIPT
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    literal_hits: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            value = node.value
            if value.startswith("/") and ".dcm" in value:
                literal_hits.append(f"{STUDY_PATH_SCRIPT}:{node.lineno}")
    # A path split across concatenations still leaves both markers in the file.
    split_path = "/media/" in source and ".dcm" in source
    assert literal_hits == []
    assert not split_path


def test_ci_workflow_rejects_notebook_outputs() -> None:
    workflow = (REPO_ROOT / CI_WORKFLOW).read_text(encoding="utf-8")
    executed = [
        line
        for line in workflow.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    checked = any("nbstripout" in line or "nb-clean" in line for line in executed)
    assert checked, f"{CI_WORKFLOW} does not run nbstripout or nb-clean"
