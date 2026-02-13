# FBTK: Forblaze Toolkit

High-performance molecular analysis and building tools powered by Rust.
Designed as a "Transparent Accelerator" for Python (ASE/RDKit) workflows with a smart, object-oriented interface.

## Features

- **🚀 High Performance**: Core logic written in Rust with parallel processing (Rayon).
- **🏗️ Intelligent Builder**: 
  - **Optimized Initial Packing**: Grid-based placement with uniform density.
  - **Polymer Synthesis**: Automatic chain generation with leaving atom support.
  - **Built-in 3D Generation**: 3D coordinate generation from SMILES handled by internal VSEPR + UFF engine.
  - **Fast Structural Relaxation**: O(N) Cell-list optimization with the FIRE algorithm.
- **🔍 Advanced Analysis**: 
  - Parallel RDF, MSD, COM, Angles, Dihedrals.
  - O(N) Neighbor List search.
- **📏 Robust Physics**: Correct handling of PBC, Triclinic cells, and Minimum Image Convention (MIC).

## Installation

```bash
pip install fbtk
```

### Prerequisites
- Python 3.8+
- NumPy

## Usage

### 1. Python Library

The Python API is designed to be minimal and high-performance.

#### RDF Analysis
```python
from ase.io import read
import fbtk

# Load trajectory (ASE list of Atoms)
traj = read('simulation.lammpstrj', index=':')

# Compute RDF using a simple query string
r, g_r = fbtk.compute_rdf(traj, query="O-H", r_max=10.0)
```

#### System Building
```python
import fbtk

# 1. Setup Builder
builder = fbtk.Builder(density=0.8)
builder.add_molecule_smiles("ethanol", count=50, smiles="CCO")

# 2. Build and Relax
system = builder.build()
builder.relax(steps=500)

# 3. Seamless export to ASE
atoms = system.to_ase()
atoms.write("system.xyz")
```

### 2. Command Line Interface (CLI)

After installation, the following commands are available in your environment:

```bash
# RDF Calculation for LAMMPS trajectory
fbtk-analyze rdf --input traj.lammpstrj --query "type 1 with type 2"

# System Building from YAML recipe
fbtk-build --recipe recipe.yaml --relax --output system.mol2
```

## Selection Query Syntax

FBTK supports intuitive strings to select atoms for analysis:

- **Element**: `"O"`, `"H"`, `"element C"`
- **Pairs (RDF)**: `"O-H"`, `"C - C"`
- **Index Range**: `"index 0:100"` (start:end)
- **Residue**: `"resname STY"`

---
(c) 2026 Forblaze Project