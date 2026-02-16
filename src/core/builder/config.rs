use serde::{Deserialize, Serialize};
use std::path::Path;
use anyhow::Result;
use std::fs;

#[derive(Debug, Serialize, Deserialize)]
pub struct Recipe {
    pub system: SystemConfig,
    pub components: Vec<ComponentConfig>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SystemConfig {
    pub density: f64,             // Target density in g/cm^3
    pub cell_shape: Option<[f64; 3]>, // Optional explicit box size [Lx, Ly, Lz]
    pub temperature: Option<f64>, // For future velocity initialization
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ComponentConfig {
    pub name: String,
    pub role: ComponentRole,
    pub input: InputSource,
    pub count: Option<usize>,     // Number of molecules (for solvents)
    pub polymer_params: Option<PolymerParams>,
}

#[derive(Debug, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum ComponentRole {
    Polymer,
    Molecule,
    Ion,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct InputSource {
    pub smiles: Option<String>,
    pub file: Option<String>,       // Path to a structure file (e.g., template.xyz)
    pub format: Option<String>,     // Optional format hint
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
#[serde(rename_all = "lowercase")]
pub enum Tacticity {
    Isotactic,
    Syndiotactic,
    Atactic,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct PolymerParams {
    pub degree: usize,            // Degree of polymerization
    pub n_chains: usize,          // Number of chains
    pub head_index: Option<usize>, // Atom index to connect (from template)
    pub tail_index: Option<usize>, // Atom index to connect (from template)
    pub head_leaving_index: Option<usize>, // Atom to remove at head connection
    pub tail_leaving_index: Option<usize>, // Atom to remove at tail connection
    pub tacticity: Option<Tacticity>,      // tacticity control (default: isotactic)
}

impl Recipe {
    pub fn from_yaml<P: AsRef<Path>>(path: P) -> Result<Self> {
        use anyhow::Context;
        let content = fs::read_to_string(&path)
            .with_context(|| format!("Failed to read recipe file: {:?}", path.as_ref()))?;
        let recipe: Recipe = serde_yaml::from_str(&content)
            .with_context(|| format!("Failed to parse YAML in recipe: {:?}\nPlease check if all required fields (system, components, role, input) are present.", path.as_ref()))?;
        Ok(recipe)
    }
}
