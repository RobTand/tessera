"""Moved: this exporter is now ``export_tessera_serving.py``.

Tessera serves its own wires (``tessera.serving``), so the checkpoint declares
``quant_method: "tessera"`` and there is no Gridbook lane to export for.  The
old name is kept only so a script or a receipt that cites it still runs.
"""
from export_tessera_serving import main  # noqa: F401

if __name__ == "__main__":
    main()
