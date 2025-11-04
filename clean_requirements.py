# smart_requirements.py
import subprocess
import sys
from pathlib import Path
import pkg_resources

REQ_FILE = "requirements.txt"
CLEAN_FILE = "requirements_clean.txt"

def run(cmd):
    """Run shell command and return output"""
    return subprocess.run(cmd, shell=True, text=True, capture_output=True)

def clean_requirements():
    """Remove duplicates, keep latest version"""
    seen = {}
    requirements = Path(REQ_FILE).read_text().splitlines()

    for line in requirements:
        pkg = line.strip()
        if not pkg or pkg.startswith("#"):
            continue
        name = pkg.split("==")[0].strip().lower()
        seen[name] = pkg  # keeps last version

    output = []
    for line in requirements:
        if not line.strip() or line.strip().startswith("#"):
            output.append(line)
        else:
            name = line.split("==")[0].strip().lower()
            if name in seen:
                output.append(seen.pop(name))

    Path(CLEAN_FILE).write_text("\n".join(output) + "\n")
    print(f"✅ Cleaned requirements saved to {CLEAN_FILE}")
    return output

def check_compatibility(pkg_line):
    """Check if package is available for Python 3.10"""
    pkg = pkg_line.split("==")[0]
    result = run(f"pip index versions {pkg}")
    if "ERROR" in result.stderr or not result.stdout.strip():
        print(f"⚠️ {pkg} might not be available on PyPI for Python 3.10.")
    else:
        print(f"✅ {pkg} looks available.")

def install_requirements():
    """Install cleaned requirements"""
    print("📦 Installing dependencies...")
    result = run(f"{sys.executable} -m pip install -r {CLEAN_FILE}")
    if result.returncode == 0:
        print("🎉 All packages installed successfully!")
    else:
        print("❌ Installation errors detected. Attempting auto-fix...\n")
        auto_fix_requirements()

# =========================
# 🔥 NEW: Auto-fix Logic
# =========================
def auto_fix_requirements():
    """Try upgrading failing packages to latest compatible versions"""
    failed_lines = []
    with open(CLEAN_FILE) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                pkg = line.split("==")[0]
                # test install pinned version individually
                test = run(f"{sys.executable} -m pip install {line}")
                if test.returncode != 0:
                    print(f"⚠️ {line} failed. Trying latest version instead...")
                    upgrade = run(f"{sys.executable} -m pip install {pkg} --upgrade")
                    if upgrade.returncode == 0:
                        print(f"✅ {pkg} upgraded successfully!")
                        failed_lines.append(f"{pkg}")  # drop version pin
                    else:
                        print(f"❌ {pkg} could not be installed at all.")
                        failed_lines.append(line)
                else:
                    print(f"✅ {line} installed fine.")

    # rewrite cleaned requirements with fixed versions
    with open(CLEAN_FILE, "w") as f:
        for line in failed_lines:
            f.write(line + "\n")
    print(f"🔄 Updated {CLEAN_FILE} with auto-fixed versions.")

    print("📦 Re-running installation with fixed requirements...")
    run(f"{sys.executable} -m pip install -r {CLEAN_FILE}")

if __name__ == "__main__":
    cleaned = clean_requirements()
    print("\n🔍 Checking Python 3.10 compatibility...")
    for line in cleaned:
        if line.strip() and not line.startswith("#"):
            check_compatibility(line)
    install_requirements()
