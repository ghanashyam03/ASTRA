import sys
import re

path = sys.argv[1]
with open(path, 'r', encoding='utf-8') as f:
    txt = f.read()

pattern = re.compile(r"from\s+[\w\.]+\s+import\s*\((.*?)\)", re.S)
errors = []
for m in pattern.finditer(txt):
    block = m.group(1)
    lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
    names = []
    for ln in lines:
        # remove trailing comma and inline comments
        ln_clean = re.sub(r"#.*", "", ln).strip()
        if ln_clean.endswith(','):
            ln_clean = ln_clean[:-1].strip()
        if not ln_clean:
            continue
        # take first token (handles aliasing unlikely here)
        token = ln_clean.split()[0]
        names.append(token)
    sorted_names = sorted(names, key=lambda s: s.lower())
    if names != sorted_names:
        errors.append((m.group(0), names, sorted_names))

if errors:
    print(f"Found {len(errors)} import blocks with unsorted members:\n")
    for blk, names, sorted_names in errors:
        print("BLOCK:\n", blk)
        print("FOUND:", names)
        print("EXPECTED:", sorted_names)
        print()
    sys.exit(1)
else:
    print("All import-member lists are case-insensitively sorted.")
    sys.exit(0)
