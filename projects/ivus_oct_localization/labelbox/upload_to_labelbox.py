"""Upload the IVUS/OCT dataset to Labelbox, create the ontology and project, and batch the data rows.

Requires: pip install labelbox ; export LABELBOX_API_KEY=...
Usage: python upload_to_labelbox.py [/media/data1/datasets/DeepCoro/labelbox_ivus_oct] [--dry-run]
Data rows are uploaded from the local mp4 files (Labelbox-hosted storage). Re-running is safe: global keys
that already exist are skipped.
"""
import json
import os
import sys

OUT = next((a for a in sys.argv[1:] if not a.startswith("--")), "/media/data1/datasets/DeepCoro/labelbox_ivus_oct")
DRY = "--dry-run" in sys.argv
HERE = os.path.dirname(os.path.abspath(__file__))
DATASET, PROJECT = "angiography-ivus-oct", "deepcoro-ivus-oct"


def build_ontology(lb, spec):
    from labelbox import Classification, MediaType, OntologyBuilder, Option
    cls = []
    for c in spec["classifications"]:
        cls.append(Classification(class_type=Classification.Type.RADIO, name=c["name"], required=c.get("required", False),
                                  scope=Classification.Scope.GLOBAL,
                                  options=[Option(value=o["value"], label=o["label"]) for o in c["options"]]))
    return lb.create_ontology(spec["name"], OntologyBuilder(classifications=cls).asdict(), media_type=MediaType.Video)


def main():
    rows = json.load(open(f"{OUT}/data_rows.json"))
    spec = json.load(open(f"{HERE}/ontology.json"))
    missing = [r["row_data"] for r in rows if not os.path.exists(r["row_data"])]
    print(f"{len(rows)} data rows, {len(missing)} missing files")
    if DRY:
        return
    import labelbox as lb_mod
    lb = lb_mod.Client(api_key=os.environ["LABELBOX_API_KEY"])

    # metadata field for study grouping
    mdo = lb.get_data_row_metadata_ontology()
    if "study_key" not in [f.name for f in mdo.custom_fields]:
        from labelbox.schema.data_row_metadata import DataRowMetadataKind
        mdo.create_schema("study_key", DataRowMetadataKind.string)
        mdo = lb.get_data_row_metadata_ontology()
    sid = next(f.uid for f in mdo.custom_fields if f.name == "study_key")
    for r in rows:
        r["metadata_fields"] = [{"schema_id": sid, "value": m["value"]} for m in r["metadata_fields"]]

    ds = next(iter(lb.get_datasets(where=lb_mod.Dataset.name == DATASET)), None) or lb.create_dataset(name=DATASET)
    existing = set()
    try:
        res = lb.get_data_row_ids_for_global_keys([r["global_key"] for r in rows])
        existing = {gk for gk, i in zip([r["global_key"] for r in rows], res["results"]) if i}
    except Exception:  # noqa: BLE001
        pass
    todo = [r for r in rows if r["global_key"] not in existing]
    print("uploading", len(todo))
    for i in range(0, len(todo), 500):
        task = ds.create_data_rows(todo[i:i + 500])
        task.wait_till_done()
        print(i, task.status, (task.errors or "")[:300] if task.errors else "")

    ont = build_ontology(lb, spec)
    proj = lb.create_project(name=PROJECT, media_type=lb_mod.MediaType.Video, description=spec["instructions"][:1000])
    proj.connect_ontology(ont)
    keys = [r["global_key"] for r in rows]
    for i in range(0, len(keys), 1000):
        proj.create_batch(f"ivus-oct-{i // 1000:02d}", global_keys=keys[i:i + 1000], priority=5)
    print("project", proj.uid)


if __name__ == "__main__":
    main()
