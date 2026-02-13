import subprocess
import sys
import os
import shutil

def run_bin(bin_name):
    # Search order:
    # 1. Same directory as this script (typical for installed wheel)
    # 2. ../target/release (typical for maturin develop --release)
    # 3. ../target/debug (typical for maturin develop)
    # 4. PATH
    
    current_dir = os.path.dirname(__file__)
    pkg_root = os.path.dirname(current_dir)
    
    candidates = [
        os.path.join(current_dir, bin_name),
        os.path.join(pkg_root, "target", "release", bin_name),
        os.path.join(pkg_root, "target", "debug", bin_name),
    ]
    
    bin_path = None
    for cand in candidates:
        if os.path.exists(cand):
            bin_path = cand
            break
            
    if not bin_path:
        bin_path = shutil.which(bin_name)
    
    if not bin_path or os.path.isdir(bin_path): # shutil.which might find the script itself
        # Final attempt: specifically look for the binary in path but avoid the script itself
        # by checking if it's a binary or just looking further.
        # But usually, if it's not in candidates, we are in trouble.
        pass
        
    if not bin_path:
        print(f"Error: {bin_name} binary not found.")
        print(f"Searched in: {candidates}")
        sys.exit(1)
        
    result = subprocess.run([bin_path] + sys.argv[1:])
    sys.exit(result.returncode)

def analyze():
    run_bin("fbtk-analyze")

def build():
    run_bin("fbtk-build")
