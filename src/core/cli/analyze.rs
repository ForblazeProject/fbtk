use clap::{Parser, Subcommand};
use std::time::Instant;
use anyhow::{Result, anyhow};
use crate::core::rdf::{compute_rdf_core, RdfParams};
use crate::core::msd::{compute_msd_core};
use crate::parsers::lammps::{parse_dump_file, traj_to_ndarray, get_atom_info};
use crate::parsers::gromacs::{GromacsTrajectory};
use crate::core::selection::{SelectionEngine, AtomInfo};
use ndarray::{Array3};

#[derive(Parser)]
#[command(name = "fbtk-analyze")]
#[command(version, about = "Forblaze Toolkit - Analyzer", long_about = "Forblaze Toolkit (c) 2026 Forblaze Project https://forblaze-works.com/")]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,

    /// Number of threads (default: 4 or RAYON_NUM_THREADS)
    #[arg(short, long, global = true)]
    pub threads: Option<usize>,
}

#[derive(Subcommand)]
pub enum Commands {
    /// Compute Radial Distribution Function (RDF)
    Rdf {
        /// Input LAMMPS dump or GROMACS gro file
        input: String,

        /// Smart Query (e.g., "type 1 with type 2" or "element O with element H")
        #[arg(short, long)]
        query: String,

        /// Maximum distance for RDF
        #[arg(long, default_value_t = 10.0)]
        rmax: f64,

        /// Number of bins
        #[arg(long, default_value_t = 200)]
        bins: usize,

        /// Output file (.dat)
        #[arg(short, long)]
        output: Option<String>,
    },

    /// Compute Mean Squared Displacement (MSD)
    Msd {
        /// Input LAMMPS dump or GROMACS gro file
        input: String,

        /// Smart Query (e.g., "type 1" or "element C")
        #[arg(short, long)]
        query: String,

        /// Maximum lag steps (0 for all)
        #[arg(long, default_value_t = 0)]
        max_lag: usize,

        /// Time step in physical units (fs)
        #[arg(long, default_value_t = 1.0)]
        dt: f64,

        /// Output file (.dat)
        #[arg(short, long)]
        output: Option<String>,
    },
}

/// Helper to load either LAMMPS or GROMACS trajectory into ndarrays.
fn load_trajectory(path: &str) -> Result<(Array3<f64>, Array3<f64>, Vec<AtomInfo>)> {
    if path.to_lowercase().ends_with(".gro") {
        let gro = GromacsTrajectory::open(path)?;
        let info = gro.get_atom_info();
        
        let mut all_pos = Vec::new();
        let mut all_cells = Vec::new();
        
        for (pos, cell) in gro {
            all_pos.push(pos);
            all_cells.push(cell);
        }
        
        if all_pos.is_empty() { return Err(anyhow!("No frames found in GRO file")); }
        
        let n_frames = all_pos.len();
        let n_atoms = all_pos[0].len();
        
        let mut pos_arr = Array3::<f64>::zeros((n_frames, n_atoms, 3));
        let mut cell_arr = Array3::<f64>::zeros((n_frames, 3, 3));
        
        for f in 0..n_frames {
            for i in 0..n_atoms {
                for d in 0..3 { pos_arr[[f, i, d]] = all_pos[f][i][d]; }
            }
            for r in 0..3 {
                for c in 0..3 { cell_arr[[f, r, c]] = all_cells[f][r][c]; }
            }
        }
        Ok((pos_arr, cell_arr, info))
    } else {
        // Assume LAMMPS dump
        let traj = parse_dump_file(path)?;
        let info = get_atom_info(&traj.frames[0]);
        let (pos, cells) = traj_to_ndarray(&traj);
        Ok((pos, cells, info))
    }
}

pub fn run_analyze_cli(args: Vec<String>) -> Result<()> {
    let cli = Cli::parse_from(args);

    // Initialize Thread Pool
    let num_threads = cli.threads.or_else(|| {
        std::env::var("RAYON_NUM_THREADS")
            .ok()
            .and_then(|s| s.parse().ok())
    }).unwrap_or(4);

    if rayon::ThreadPoolBuilder::new().num_threads(num_threads).build_global().is_err() {
        // Already initialized
    }

    match &cli.command {
        Commands::Rdf { input, query, rmax, bins, output } => {
            println!("Reading trajectory: {}...", input);
            let (pos, cells, info) = load_trajectory(input)?;
            let engine = SelectionEngine::new(info);
            let (idx_a, idx_b) = engine.select_pair(query)?;

            println!("Selected Group A: {} atoms, Group B: {} atoms", idx_a.len(), idx_b.len());

            let params = RdfParams { r_max: *rmax, n_bins: *bins };
            
            println!("Computing RDF (Frames: {}, Threads: {})...", pos.shape()[0], num_threads);
            let start = Instant::now();
            let result = compute_rdf_core(pos.view(), cells.view(), &idx_a, &idx_b, params);
            println!("Finished in {:.2?}", start.elapsed());

            if let Some(out_path) = output {
                use std::io::Write;
                let mut f = std::fs::File::create(out_path)?;
                writeln!(f, "# r(A) g(r)")?;
                for (r, g) in result.r_axis.iter().zip(result.g_r.iter()) {
                    writeln!(f, "{:.4} {:.4}", r, g)?;
                }
                println!("Saved to {}", out_path);
            }
        },

        Commands::Msd { input, query, max_lag, dt, output } => {
            println!("Reading trajectory: {}...", input);
            let (pos, cells, info) = load_trajectory(input)?;
            let engine = SelectionEngine::new(info);
            let idx = engine.select(query)?;

            println!("Selected: {} atoms", idx.len());

            println!("Computing MSD (Frames: {}, Threads: {})...", pos.shape()[0], num_threads);
            let start = Instant::now();
            let result = compute_msd_core(pos.view(), cells.view(), &idx, *max_lag, *dt);
            println!("Finished in {:.2?}", start.elapsed());

            if let Some(out_path) = output {
                use std::io::Write;
                let mut f = std::fs::File::create(out_path)?;
                writeln!(f, "# time(fs) msd(A^2)")?;
                for (t, m) in result.time.iter().zip(result.msd_total.iter()) {
                    writeln!(f, "{:.4} {:.4}", t, m)?;
                }
                println!("Saved to {}", out_path);
            }
        }
    }
    
    Ok(())
}