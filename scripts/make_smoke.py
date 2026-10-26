"""Write a one-instance static set and a one-instance dynamic set.

WorldScore's loader looks for ``$DATA_PATH/WorldScore-Dataset/<split>/<name>``.
The full benchmark is 2000 static instances and 1000 dynamic ones. This smoke
set is the first record of each, which is what the W7900 run evaluated.
"""

import json
from pathlib import Path

root = Path("/workspace/data/WorldScore-Dataset")
for split in ("static", "dynamic"):
    data = json.loads((root / split / f"{split}.json").read_text())
    out = root / split / "smoke.json"
    out.write_text(json.dumps([data[0]], indent=2))
    print(split, out, data[0].get("image"))
