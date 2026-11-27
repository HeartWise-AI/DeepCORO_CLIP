"""Append verified labels: python add_labels.py <candidates.csv> <task> <round> <pos ids> <neg ids> [vessel map id:V,...]"""
import sys

import pandas as pd

D = "projects/ivus_oct_localization/data"
c = pd.read_csv(sys.argv[1]).set_index("cid")
task, rnd = sys.argv[2], int(sys.argv[3])
pos = [int(x) for x in sys.argv[4].split(",") if x]
neg = [int(x) for x in sys.argv[5].split(",") if x]
vmap = dict(kv.split(":") for kv in sys.argv[6].split(",")) if len(sys.argv) > 6 and sys.argv[6] else {}
new = pd.concat([c.loc[pos].assign(label=1), c.loc[neg].assign(label=0)]).reset_index()
new["vessel_visual"] = new.cid.astype(str).map(vmap)
new["task"], new["round"] = task, rnd
old = pd.read_csv(f"{D}/instance_labels.csv")
out = pd.concat([old, new[old.columns]]).drop_duplicates("FileName", keep="last")
out.to_csv(f"{D}/instance_labels.csv", index=False)
print(out.groupby(["task", "label"]).size())
