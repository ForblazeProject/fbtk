use clap::Parser;
use crate::core::builder::config::Recipe;
use crate::core::builder::model::Builder;
use std::path::PathBuf;
use anyhow::{Result, anyhow};

#[derive(Parser, Debug)]
#[command(name = "fbtk-build")]
#[command(version, about = "Forblaze Toolkit - Builder", long_about = "Forblaze Toolkit (c) 2026 Forblaze Project https://forblaze-works.com/")]
pub struct Args {
    /// Path to the recipe YAML file
    #[arg(short, long)]
    pub recipe: PathBuf,

    /// Output file path (e.g., system.mol2)
    #[arg(short, long)]
    pub output: Option<PathBuf>,

    /// Run quick relaxation after building
    #[arg(long, default_value_t = false)]
    pub relax: bool,

    /// Number of threads (default: 4 or RAYON_NUM_THREADS)
    #[arg(short, long)]
    pub threads: Option<usize>,
}

pub fn run_build_cli(args: Vec<String>) -> Result<()> {
    let args = Args::parse_from(args);

    // 1. Thread Pool Initialization
    let num_threads = args.threads.or_else(|| {
        std::env::var("RAYON_NUM_THREADS")
            .ok()
            .and_then(|s| s.parse().ok())
    }).unwrap_or(4);

    if rayon::ThreadPoolBuilder::new().num_threads(num_threads).build_global().is_err() {
        // Already initialized, which is fine
    }

    println!("Forblaze Toolkit (c) 2026 Forblaze Project https://forblaze-works.com/");
    println!("--- fbtk-build ---");
    println!("Threads: {}", num_threads);
    println!("Reading recipe from {:?}", args.recipe);
    let mut recipe = Recipe::from_yaml(&args.recipe)?;
    
    // 2. Configuration Summary & Strict Validation
    println!("\nSystem Summary:");
    println!("  Target Density: {} g/cm^3", recipe.system.density);
    if let Some(shape) = recipe.system.cell_shape {
        println!("  Cell Shape: [{:.2}, {:.2}, {:.2}] A", shape[0], shape[1], shape[2]);
    }
    println!("Components:");
    for comp in &recipe.components {
        match comp.role {
            crate::core::builder::config::ComponentRole::Polymer => {
                let p = comp.polymer_params.as_ref().ok_or_else(|| {
                    anyhow!("Component '{}' is marked as 'polymer' but missing required 'polymer_params' key in YAML.", comp.name)
                })?;
                println!("  - [{}] {}: degree={}, n_chains={}, tacticity={:?}", 
                    "Polymer", comp.name, p.degree, p.n_chains, p.tacticity.as_ref().unwrap_or(&crate::core::builder::config::Tacticity::Isotactic));
            },
            _ => {
                println!("  - [{:?}] {}: count={}", 
                    comp.role, comp.name, comp.count.unwrap_or(1));
            }
        }
    }
    println!("");

    let mut builder = Builder::new();
    
    // 3. Load Templates
    let recipe_dir = args.recipe.parent().unwrap_or(std::path::Path::new("."));
    
    for comp in &mut recipe.components {
        if let Some(file_path) = &comp.input.file {
            let full_path = recipe_dir.join(file_path);
            println!("Loading template for {}: {:?}", comp.name, full_path);
            builder.load_template_file(&comp.name, full_path.to_str().unwrap_or(""))?;
        } else if let Some(smiles) = &comp.input.smiles {
            println!("Generating template from SMILES for {}: {}", comp.name, smiles);
            let tmpl = crate::core::builder::smiles::parse_smiles(smiles)?;
            
            // Resolve indices for polymers if needed
            if comp.role == crate::core::builder::config::ComponentRole::Polymer {
                let p = comp.polymer_params.as_mut().unwrap(); // Already validated above
                
                if p.head_index.is_none() || p.tail_index.is_none() {
                    let (h, t, hl, tl) = crate::core::builder::smiles::resolve_polymer_indices(&tmpl, p.head_index, p.tail_index)?;
                    if p.head_index.is_none() { p.head_index = Some(h); }
                    if p.tail_index.is_none() { p.tail_index = Some(t); }
                    if p.head_leaving_index.is_none() { p.head_leaving_index = hl; }
                    if p.tail_leaving_index.is_none() { p.tail_leaving_index = tl; }
                    println!("  Resolved polymer indices for {}: head={}, tail={}, h_leaving={:?}, t_leaving={:?}", 
                        comp.name, h, t, hl, tl);
                }
            }
            
            builder.add_template(comp.name.clone(), tmpl);
        }
    }

    builder.set_recipe(recipe);
    
    println!("\nBuilding system...");
    builder.build()?;
    
    let system = builder.system.as_ref().ok_or_else(|| anyhow!("Failed to build system"))?;
    println!("System built successfully.");
    println!("  Atoms: {}", system.atoms.len());
    println!("  Bonds: {}", system.bonds.len());
    
    if args.relax {
        println!("Relaxing system (Default parameters)...");
        builder.relax(None, None)?;
    }

    if let Some(out_path) = args.output {
        let ext = out_path.extension().and_then(|s: &std::ffi::OsStr| s.to_str()).unwrap_or("");
        match ext {
            "mol2" => {
                let system = builder.system.as_ref().unwrap();
                let content = crate::parsers::mol2::write_mol2(system);
                std::fs::write(&out_path, content)?;
                println!("Saved to {:?}", out_path);
            }
            _ => {
                return Err(anyhow!("Unsupported output format: {}. Use .mol2", ext));
            }
        }
    }
    
    Ok(())
}
