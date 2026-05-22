#!/usr/bin/bash

# cd ..

# sudo apt update && sudo apt install build-essential libpq-dev libssl-dev openssl libffi-dev sqlite3 libsqlite3-dev libbz2-dev zlib1g-dev libxerces-c-dev libfox-1.6-dev libgdal-dev libproj-dev libgl2ps-dev git g++ cmake

# m=0 && while wget -q --method=HEAD https://www.python.org/ftp/python/3.7.$(( $m + 1 ))/Python-3.7.$(( $m + 1 )).tar.xz; do m=$(( $m + 1 )); done && wget https://www.python.org/ftp/python/3.7.$m/Python-3.7.$m.tar.xz && tar xvf Python-3.7.$m.tar.xz && cd Python-3.7.$m && ./configure && make && make altinstall && cd .. && rm -rv Python-3.7.$m.tar.xz Python-3.7.$m

# mkdir venv && python3.7 -m venv venv/

# source venv/bin/activate

# export TMPDIR='/var/tmp'
# pip3 install gym torch tensorboard 'msgpack==1.0.2' six numpy sumolib traci matplotlib wheel --no-cache-dir

# deactivate


cd venv/ && git clone --recursive https://github.com/eclipse/sumo && rm -rv $(find sumo/ -iname "*.git*")
mkdir sumo/build_config/cmake-build && cd sumo/build_config/cmake-build
cmake ../..
make -j$(nproc)

exit

#!/usr/bin/env bash
set -e  # exit if any command fails

# Go to project root (adjust if needed)
cd ..

# Install system dependencies (needed for compiling Python libs & SUMO)
sudo apt update && sudo apt install -y \
    build-essential libpq-dev libssl-dev openssl libffi-dev sqlite3 \
    libsqlite3-dev libbz2-dev zlib1g-dev libxerces-c-dev \
    libfox-1.6-dev libgdal-dev libproj-dev libgl2ps-dev git g++ cmake

# Ensure conda is available
if ! command -v conda &> /dev/null; then
    echo "❌ Conda not found. Please install Miniconda or Anaconda first."
    exit 1
fi

# Create or update the conda environment from environment.yml
conda env create -f environment.yml --name venv --force

# Enable conda commands inside bash
source "$(conda info --base)/etc/profile.d/conda.sh"

# Activate environment
conda activate venv

echo "Conda environment 'venv' created and all dependencies installed."
echo " Run: conda activate venv"

# #!/usr/bin/env bash
# set -e  # exit if any command fails

# # Go to project root (adjust if needed)
# cd ..

# # Install system dependencies (needed for compiling Python libs & SUMO)
# sudo apt update && sudo apt install -y \
#     build-essential libpq-dev libssl-dev openssl libffi-dev sqlite3 \
#     libsqlite3-dev libbz2-dev zlib1g-dev libxerces-c-dev \
#     libfox-1.6-dev libgdal-dev libproj-dev libgl2ps-dev git g++ cmake

# # Ensure conda is available
# if ! command -v conda &> /dev/null; then
#     echo "❌ Conda not found. Please install Miniconda or Anaconda first."
#     exit 1
# fi

# # Create or update the conda environment from environment.yml
# conda env create -f environment.yml --name venv --force

# # Enable conda commands inside bash
# source "$(conda info --base)/etc/profile.d/conda.sh"

# # Activate environment
# conda activate venv

# echo "✅ Conda environment 'venv' created and all dependencies installed."
# echo "👉 Run: conda activate venv"



