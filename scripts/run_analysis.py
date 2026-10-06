r"""Execute analysis.ipynb with this interpreter and retain all cell outputs.

Usage: .venv\Scripts\python.exe scripts/run_analysis.py
No network calls are made by the notebook. Input datasets must already exist.
"""
import os
import sys
from pathlib import Path

import nbformat
from jupyter_client import KernelManager
from nbclient import NotebookClient


def main():
    root = Path(__file__).resolve().parents[1]
    runtime = root / "data/processed/.jupyter_runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    os.environ["JUPYTER_RUNTIME_DIR"] = str(runtime)
    os.environ["IPYTHONDIR"] = str(runtime / "ipython")
    path = root / "notebooks/analysis.ipynb"
    notebook = nbformat.read(path, as_version=4)
    manager = KernelManager(kernel_name="python3")
    manager.kernel_spec.argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]

    def log_cell(cell, cell_index, **kwargs):
        if cell.cell_type == "code":
            print(f"Executing cell {cell_index + 1}/{len(notebook.cells)}", flush=True)

    client = NotebookClient(
        notebook, km=manager, timeout=900,
        resources={"metadata": {"path": str(root)}}, on_cell_start=log_cell,
    )
    try:
        client.execute()
    finally:
        nbformat.write(notebook, path)
        if manager.has_kernel:
            manager.shutdown_kernel(now=True)
    print(f"Executed notebook saved: {path}", flush=True)


if __name__ == "__main__":
    main()
