import sys
from .fbtk import run_analyze_cli, run_build_cli

def analyze():
    # We pass sys.argv but replace the first element with the command name 
    # so Clap is happy regardless of how the script was invoked.
    args = list(sys.argv)
    args[0] = "fbtk-analyze"
    run_analyze_cli(args)

def build():
    args = list(sys.argv)
    args[0] = "fbtk-build"
    run_build_cli(args)