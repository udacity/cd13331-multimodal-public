#!/usr/bin/env python3
"""
Update Jupyter notebook kernels to use local .venv interpreters.
This script finds all demo.ipynb files and updates their metadata
to point to the .venv/bin/python in the same directory.
"""

import json
from pathlib import Path


def find_notebooks(root_dir="."):
    """Find all demo.ipynb files in subdirectories."""
    root_path = Path(root_dir)
    notebooks = list(root_path.glob("*/demo.ipynb"))
    return notebooks


def create_vscode_settings(notebook_dir):
    """Create .vscode/settings.json with correct interpreter path."""
    vscode_dir = notebook_dir / ".vscode"
    vscode_dir.mkdir(exist_ok=True)
    
    settings_file = vscode_dir / "settings.json"
    
    # Use relative path from the notebook directory
    settings = {
        "python.defaultInterpreterPath": ".venv/bin/python"
    }
    
    with open(settings_file, 'w', encoding='utf-8') as f:
        json.dump(settings, f, indent=2)
        f.write('\n')
    
    print(f"  → Created {settings_file}")


def update_notebook_kernel(notebook_path):
    """Update the notebook to use local .venv interpreter."""
    notebook_dir = notebook_path.parent
    venv_path = notebook_dir / ".venv"
    python_path = venv_path / "bin" / "python"
    
    # Check if .venv/bin/python exists
    if not python_path.exists():
        print(f"⚠️  Skipping {notebook_path}: {python_path} not found")
        return False
    
    # Read the notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    # Update the kernel metadata (minimal, VS Code will recognize it)
    if 'metadata' not in notebook:
        notebook['metadata'] = {}
    
    # Set a generic Python 3 kernelspec
    notebook['metadata']['kernelspec'] = {
        'display_name': 'Python 3',
        'language': 'python',
        'name': 'python3'
    }
    
    # Write back the notebook
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)
        f.write('\n')  # Add trailing newline
    
    # Create VS Code settings for this directory
    create_vscode_settings(notebook_dir)
    
    print(f"✓ Updated {notebook_path} -> .venv/bin/python")
    return True


if __name__ == "__main__":
    import sys
    
    root_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    
    print("=" * 60)
    print("Updating notebook interpreters to local .venv")
    print("=" * 60)
    
    notebooks = find_notebooks(root_dir)
    
    if not notebooks:
        print("No demo.ipynb files found!")
        sys.exit(1)
    
    print(f"Found {len(notebooks)} notebook(s)\n")
    
    updated = 0
    for notebook in notebooks:
        if update_notebook_kernel(notebook):
            updated += 1
    
    print("\n" + "=" * 60)
    print(f"Summary: Updated {updated}/{len(notebooks)} notebooks")
    print("=" * 60)
    print("\nEach folder now has a .vscode/settings.json pointing to .venv/bin/python")
    print("VS Code should automatically use the correct interpreter.")