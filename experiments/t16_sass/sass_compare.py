"""Compare two cuobjdump -sass listings function by function.

Usage: python3 sass_compare.py OLD.sass NEW.sass

Functions are keyed by demangled name (the anonymous namespace's file hash is
dropped by demangling). A function's body is its instruction text with the
addresses and encodings stripped and mangled symbol operands folded, so two
builds of the same source compare equal. Prints one line per function that
differs, is missing, or is new, then a summary line:
``identical N of M old; K new``.
"""
import re
import subprocess
import sys


def functions(path):
    out, cur, body = {}, None, []
    with open(path) as handle:
        for line in handle:
            m = re.match(r"\s*Function : (\S+)", line)
            if m:
                if cur:
                    out[cur] = body
                cur, body = m.group(1), []
                continue
            if cur is None:
                continue
            m = re.match(r"\s*/\*[0-9a-f]{4,}\*/\s*(.*?)\s*;", line)
            if m:
                body.append(re.sub(r"_Z\w+", "SYM", m.group(1)))
    if cur:
        out[cur] = body
    return out


def demangled(names):
    text = subprocess.run(["c++filt"], input="\n".join(names), capture_output=True, text=True,
                          check=True).stdout.split("\n")
    return dict(zip(names, text))


def main(old_path, new_path):
    a, b = functions(old_path), functions(new_path)
    da, db = demangled(list(a)), demangled(list(b))
    old = {da[n]: v for n, v in a.items()}
    new = {db[n]: v for n, v in b.items()}
    same = 0
    for name in sorted(old):
        if name not in new:
            print("MISSING", name)
        elif old[name] == new[name]:
            same += 1
        else:
            print("DIFF", name, len(old[name]), len(new[name]))
    for name in sorted(set(new) - set(old)):
        print("NEW", name, len(new[name]))
    print("identical", same, "of", len(old), "old;", len(new), "new")
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:3]))
