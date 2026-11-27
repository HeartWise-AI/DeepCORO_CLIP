"""Re-upload data rows missing from the Labelbox dataset and add them to the existing project as a new batch.
Usage: LABELBOX_API_KEY=... python retry_failed.py <project_id> [package_dir]"""
import json, os, sys, time
import labelbox as lb_mod
from labelbox.schema.data_row_metadata import DataRowMetadataKind  # noqa: F401

PID = sys.argv[1]
OUT = sys.argv[2] if len(sys.argv) > 2 else "/media/data1/datasets/DeepCoro/labelbox_ivus_oct"
lb = lb_mod.Client(api_key=os.environ["LABELBOX_API_KEY"])
rows = json.load(open(f"{OUT}/data_rows.json"))
sid = next(f.uid for f in lb.get_data_row_metadata_ontology().custom_fields if f.name == "study_key")
ds = next(iter(lb.get_datasets(where=lb_mod.Dataset.name == "angiography-ivus-oct")))
proj = lb.get_project(PID)
for attempt in range(4):
    keys = [r["global_key"] for r in rows]
    res = lb.get_data_row_ids_for_global_keys(keys)
    missing = [r for r, i in zip(rows, res["results"]) if not i]
    print("attempt", attempt, "missing", len(missing), flush=True)
    if not missing:
        break
    for r in missing:
        r["metadata_fields"] = [{"schema_id": sid, "value": m["value"]} for m in r["metadata_fields"]]
    for i in range(0, len(missing), 50):
        t = ds.create_data_rows(missing[i:i + 50]); t.wait_till_done()
        print(" chunk", i, t.status, len(t.errors or []), flush=True)
    time.sleep(30)
# make sure every uploaded key is in a batch of this project
in_proj = set()
for b in proj.batches():
    pass
added = [r["global_key"] for r, i in zip(rows, lb.get_data_row_ids_for_global_keys([r["global_key"] for r in rows])["results"]) if i]
try:
    proj.create_batch(f"ivus-oct-retry-{int(time.time())}", global_keys=added, priority=5)
except Exception as e:  # keys already batched raise / are skipped
    print("batch note:", str(e)[:300])
print("total rows in dataset:", len(added), "/", len(rows))
