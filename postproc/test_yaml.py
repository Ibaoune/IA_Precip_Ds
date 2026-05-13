import yaml
with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)
custom = cfg.get("custom_limits", {})
print("mean in custom?", "mean" in custom)
print(custom.get("mean", {}))

match = None
base_name = "mean"
for k in sorted(custom.keys(), key=len, reverse=True):
    if k in base_name:
        match = k
        break
print("match:", match)
if match and "spatial" in custom[match]:
    print("limits:", custom[match]["spatial"])
else:
    print("no limits found")
