"""Apply the explicit source patch files before the plugin install."""
import argparse
from pathlib import Path
import runpy


def apply_patches(names, root=None):
    root = Path(root) if root is not None else Path(__file__).with_name("patches")
    for name in names:
        if Path(name).name != name or not name.endswith(".py"):
            raise ValueError("Patch files must use local Python file names.")
        runpy.run_path(str(root / name), run_name="__main__")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patch-files", default="")
    args = parser.parse_args()
    apply_patches([name for name in args.patch_files.split(",") if name])


if __name__ == "__main__":
    main()
