use anyhow::Result;
use fbtk::core::cli::build::run_build_cli;

fn main() -> Result<()> {
    run_build_cli(std::env::args().collect())
}
