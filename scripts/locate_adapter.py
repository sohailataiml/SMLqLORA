import json, os
from pathlib import Path

RUN = "socratic-v1-n600"

# Everywhere an adapter could plausibly be.
SEARCH_ROOTS = [
    Path(f"outputs/{RUN}"),
    Path("outputs"),
    Path(f"/content/drive/MyDrive/socratic-debug-tutor/{RUN}"),
    Path("/content/drive/MyDrive/socratic-debug-tutor"),
]

try:
    from google.colab import drive
    if not Path("/content/drive/MyDrive").exists():
        drive.mount("/content/drive")
except Exception as exc:
    print(f"(Drive not mounted: {exc})")

print("=" * 78)
print("WHAT IS ACTUALLY ON DISK")
print("=" * 78)
for root in SEARCH_ROOTS:
    print(f"\n### {root}  exists={root.exists()}")
    if not root.exists():
        continue
    for path in sorted(root.rglob("*"))[:60]:
        if path.is_file():
            print(f"   {path.relative_to(root)}  ({path.stat().st_size/2**20:.2f} MiB)")
        else:
            print(f"   {path.relative_to(root)}/")

print()
print("=" * 78)
print("ADAPTERS FOUND (a directory containing adapter_config.json)")
print("=" * 78)
found = []
for root in SEARCH_ROOTS:
    if not root.exists():
        continue
    for cfg in root.rglob("adapter_config.json"):
        weights = [p for p in cfg.parent.iterdir()
                   if p.suffix in (".safetensors", ".bin") and "adapter" in p.name]
        size = sum(p.stat().st_size for p in weights) / 2**20
        found.append((cfg.parent, weights, size))
        print(f"\n  {cfg.parent}")
        print(f"    weights : {[p.name for p in weights]}  ({size:.1f} MiB)")
        try:
            meta = json.loads(cfg.read_text())
            print(f"    base    : {meta.get('base_model_name_or_path')}")
            print(f"    r/alpha : {meta.get('r')}/{meta.get('lora_alpha')}")
            print(f"    targets : {sorted(meta.get('target_modules') or [])}")
        except Exception as exc:
            print(f"    !! unreadable adapter_config.json: {exc}")

print()
print("=" * 78)
if not found:
    print("NO ADAPTER ANYWHERE.")
    print()
    print("Before concluding the checkpoint is lost, check the training log:")
    print("   results/training/socratic-v1-n600.log")
    print("   outputs/socratic-v1-n600/checkpoint_metadata.json  ('completed' field)")
    print()
    print("If training never reached trainer.save_model(), retraining IS justified")
    print("- that is the one case where it is. Otherwise the files are misplaced,")
    print("not missing.")
else:
    usable = [f for f in found if f[1]]
    print(f"{len(found)} adapter dir(s), {len(usable)} with weights.")
    if usable:
        best = max(usable, key=lambda f: f[2])
        print(f"\nUSE THIS ONE:\n    ADAPTER = Path(r'{best[0]}')")
        print("\nSet that in the notebook and re-run the checkpoint-validation cell.")

# Did training actually finish?
meta_path = Path(f"outputs/{RUN}/checkpoint_metadata.json")
if meta_path.exists():
    meta = json.loads(meta_path.read_text())
    print(f"\ntraining metadata: completed={meta.get('completed')} "
          f"train_size={meta.get('dataset_train_size')} "
          f"source_hash={str(meta.get('source_dataset_hash'))[:16]}")
else:
    print(f"\n(no {meta_path} - the training run's metadata is missing too)")
