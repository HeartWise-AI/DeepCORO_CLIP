"""Set a top-level scalar key in a YAML config in place, keeping the rest of the
file (comments, `!!type` tags, ordering) untouched.

Usage: python scripts/set_yaml_key.py CONFIG KEY VALUE [KEY VALUE ...]

VALUE is written verbatim (quote strings that contain ':' or '#'). A missing key
is appended at the end of the file. Used by scripts/run_echojepa_benchmark.sh
because the `yq` on some hosts is the Python jq wrapper, not mikefarah/yq.
"""

import re
import sys


def set_key(text: str, key: str, value: str) -> str:
    pat = re.compile(rf"^{re.escape(key)}:[^\n]*$", re.MULTILINE)
    line = f"{key}: {value}"
    if pat.search(text):
        return pat.sub(line, text, count=1)
    return text.rstrip("\n") + "\n" + line + "\n"


def main() -> None:
    if len(sys.argv) < 4 or (len(sys.argv) - 2) % 2:
        sys.exit(__doc__)
    path = sys.argv[1]
    with open(path) as f:
        text = f.read()
    for key, value in zip(sys.argv[2::2], sys.argv[3::2]):
        text = set_key(text, key, value)
    with open(path, "w") as f:
        f.write(text)


if __name__ == "__main__":
    main()
