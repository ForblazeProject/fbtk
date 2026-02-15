use anyhow::Result;
use fbtk::core::cli::analyze::run_analyze_cli;

fn main() -> Result<()> {
    run_analyze_cli(std::env::args().collect())
}