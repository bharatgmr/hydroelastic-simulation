# Source this: `source env.sh` — activates the Isaac Sim venv for this project.
_HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$_HERE/.venv/bin/activate"
export OMNI_KIT_ACCEPT_EULA=YES   # accepts the NVIDIA Omniverse EULA non-interactively
unset _HERE
