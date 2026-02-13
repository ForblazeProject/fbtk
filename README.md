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
  - Parallel RDF, MSD, COM (Center of Mass), Angles, Dihedrals.
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

### 1. System Building

Build and relax a complex molecular system with just a few lines of code.

```python
import fbtk

# 1. Setup Builder
builder = fbtk.Builder(density=0.8)
builder.add_molecule_smiles("ethanol", count=50, smiles="CCO")

# 2. Build and Relax
system = builder.build()
system.relax(steps=500)

# 3. Export to ASE
atoms = system.to_ase()
atoms.write("system.xyz")
```

### 2. RDF Analysis

Fast analysis of large trajectories using smart selection queries.

```python
from ase.io import read
import fbtk

# Load trajectory (ASE list of Atoms)
traj = read('simulation.lammpstrj', index=':')

# Compute RDF using a simple query string
r, g_r = fbtk.compute_rdf(traj, query="O-H", r_max=10.0)
```

---

### 3. Command Line Interface (CLI)

FBTK provides standalone CLI tools for batch processing.

#### fbtk-build: Build from Recipe
```bash
# Run building and relaxation from a YAML recipe
fbtk-build --recipe recipe.yaml --relax --output system.mol2
```

Example `recipe.yaml`:
```yaml
system:
  density: 0.8
  cell_shape: [20.0, 20.0, 20.0]
components:
  - name: "ethanol"
    role: "molecule"
    input:
      smiles: "CCO"
    count: 50
```

#### fbtk-analyze: Analyze Trajectory
```bash
# Compute RDF for a LAMMPS trajectory
fbtk-analyze rdf --input traj.lammpstrj --query "type 1 with type 2"
```

## Selection Query Syntax

FBTK supports intuitive strings to select atoms for analysis:

- **Element**: `"O"`, `"H"`, `"element C"`
- **Pairs (RDF)**: `"O-H"`, `"C - C"`
- **Index Range**: `"index 0:100"` (start:end)
- **Residue**: `"resname STY"`

---
### Author
**Forblaze Project**  
Website: [https://forblaze-works.com/](https://forblaze-works.com/)

