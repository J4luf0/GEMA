# Approximate commands to install AdaptiveCpp (WSL + CUDA)

## Prepare

### On host (Windows)

1. Have Nvidia GPU.

2. Have newest or at least newer drivers for your Nvidia GPU.

### In WSL (Ubuntu)

For cuda toolkit to be installable. Follow commands in link. But probably better to replace 13.2.1 with 12.2.0 for better stability with AdaptiveCpp.

https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=WSL-Ubuntu&target_version=2.0&target_type=deb_local

**More close to this:**

More modern approach:
```
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb

sudo dpkg -i cuda-keyring_1.1-1_all.deb

apt update
```

or manual if previous doesnt work:
```
wget https://developer.download.nvidia.com/compute/cuda/12.2.0/local_installers/cuda-repo-wsl-ubuntu-12-2-local_12.2.0-1_amd64.deb

wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-wsl-ubuntu.pin

sudo mv cuda-wsl-ubuntu.pin /etc/apt/preferences.d/cuda-repository-pin-600

sudo dpkg -i cuda-repo-*.deb    # probably this isntead of star: wsl-ubuntu-12-2-local_12.2.0-1_amd64
# next command should be outputted with correct number from previous command
sudo cp /var/cuda-repo-*/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt update
```

**Now install needed packages**

Minimum required (with CUDA toolkit):

```
sudo apt install -y \
  build-essential \
  clang-18 \
  clang-tools-18 \
  llvm-18-dev \
  libclang-18-dev \
  lld \
  cmake \
  ninja-build \
  git \
  python3 \
  libboost-all-dev \
  libhwloc-dev \
  libomp-dev \
  cuda-compiler-12-2 \
  cuda-libraries-12-2 \
  cuda-keyring
```

Adding nvcc (cuda compiler) to PATH:

```
nano ~/.bashrc
```

Add this to top:

```
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

And reload:

```
source ~/.bashrc
```


## Install:

1. Clone **AdaptiveCpp** repo.

```
git clone https://github.com/AdaptiveCpp/AdaptiveCpp.git
cd AdaptiveCpp
```

2. Create /build directory in AdaptiveCpp repository top directory and get in it.

```
mkdir build
cd build
```

3. Instalation itself.

```
cmake ..  -G Ninja  -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DWITH_CUDA_BACKEND=ON \
    -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda
```

```
ninja
```

```
sudo ninja install
```

### Uninstall:

```
sudo ninja uninstall
```

## Use in project

Refer to  **CMakeLists.txt** and **make.sh**.
