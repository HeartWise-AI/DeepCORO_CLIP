#!/bin/bash
# Monitor wandb run 7qrv8bj2 every 2 hours, print val MAE + pearson_r summary
# Compare against tough-fire-2 (w4q1oqdu) baseline

RUN_ID="7qrv8bj2"
BASELINE_ID="w4q1oqdu"
PROJECT="mhi_ai/DeepCORO_stenosis_weighted_distal"
INTERVAL=7200  # 2 hours in seconds

while true; do
    echo "========================================"
    echo "$(date) — Checking run ${RUN_ID}"
    echo "========================================"

    .venv/bin/python -c "
import wandb, numpy as np
api = wandb.Api()

new_run = api.run('${PROJECT}/${RUN_ID}')
old_run = api.run('${PROJECT}/${BASELINE_ID}')

segments = [
    'prox_rca', 'mid_rca', 'dist_rca', 'pda', 'posterolateral',
    'left_main', 'prox_lad', 'mid_lad', 'dist_lad', 'D1', 'D2',
    'prox_lcx', 'mid_lcx', 'dist_lcx', 'om1', 'om2', 'bx', 'lvp'
]

mae_keys = [f'val/{s}_stenosis_mae' for s in segments]
pearson_keys = [f'val/{s}_stenosis_pearson_r' for s in segments]

new_hist = new_run.history(keys=mae_keys + pearson_keys + ['val/main_loss', '_step'], samples=500)
old_hist = old_run.history(keys=mae_keys + pearson_keys + ['val/main_loss', '_step'], samples=500)

if len(new_hist) == 0:
    print('New run has no val data yet. Still training epoch 1?')
else:
    new_latest = new_hist.iloc[-1]
    old_best_idx = old_hist['val/main_loss'].idxmin()
    old_best = old_hist.iloc[old_best_idx]

    print(f'New run: step {int(new_latest[\"_step\"])}, latest epoch val results')
    print(f'Old run (tough-fire-2): best val loss epoch')
    print()
    print(f'{\"Segment\":<25} {\"New MAE\":>10} {\"Old MAE\":>10} {\"New Pearson\":>12} {\"Old Pearson\":>12}')
    print('-' * 72)

    new_maes, old_maes, new_rs, old_rs = [], [], [], []
    for s in segments:
        mk = f'val/{s}_stenosis_mae'
        pk = f'val/{s}_stenosis_pearson_r'
        nm = new_latest.get(mk, float('nan'))
        om = old_best.get(mk, float('nan'))
        nr = new_latest.get(pk, float('nan'))
        orr = old_best.get(pk, float('nan'))
        if not np.isnan(nm): new_maes.append(nm)
        if not np.isnan(om): old_maes.append(om)
        if not np.isnan(nr): new_rs.append(nr)
        if not np.isnan(orr): old_rs.append(orr)
        print(f'{s:<25} {nm:>10.4f} {om:>10.4f} {nr:>12.4f} {orr:>12.4f}')

    print('-' * 72)
    print(f'{\"MEAN\":<25} {np.mean(new_maes):>10.4f} {np.mean(old_maes):>10.4f} {np.mean(new_rs):>12.4f} {np.mean(old_rs):>12.4f}')
    print(f'New val/main_loss: {new_latest.get(\"val/main_loss\", \"N/A\")}')
    print(f'Old best val/main_loss: {old_best.get(\"val/main_loss\", \"N/A\")}')
    print()
    print(f'Run state: {new_run.state}')

print()
" 2>/dev/null

    # Check if run is finished
    RUN_STATE=$(.venv/bin/python -c "
import wandb
api = wandb.Api()
run = api.run('${PROJECT}/${RUN_ID}')
print(run.state)
" 2>/dev/null)

    if [ "$RUN_STATE" = "finished" ] || [ "$RUN_STATE" = "crashed" ] || [ "$RUN_STATE" = "failed" ]; then
        echo "Run ended with state: $RUN_STATE"
        break
    fi

    echo "Next check in 2 hours..."
    sleep $INTERVAL
done
