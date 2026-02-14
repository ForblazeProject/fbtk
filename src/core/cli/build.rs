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
        eprintln!("Warning: Failed to initialize thread pool (already initialized?)");
    }

    println!("Forblaze Toolkit (c) 2026 Forblaze Project https://forblaze-works.com/");
    println!("--- fbtk-build ---");
    println!("Threads: {}", num_threads);
    println!("Reading recipe from {:?}", args.recipe);
    let recipe = Recipe::from_yaml(&args.recipe)?;
    
    let mut builder = Builder::new();
    builder.set_recipe(recipe);
    
    println!("Building system...");
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
