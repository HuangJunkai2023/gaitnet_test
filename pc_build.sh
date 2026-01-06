#!/usr/bin/env bash
set -e

# Use GCC 9 to match the libraries compiled with GCC < 10
export CC=/usr/bin/gcc-9
export CXX=/usr/bin/g++-9

CONDA_PREFIX=$HOME/miniconda3/envs/gaitnet
ENVDIR=${ENVDIR:-$HOME/pkgenv}

# Clean build directory to avoid GCC version conflicts
rm -rf build
mkdir -p build
pushd build

cmake \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="${CONDA_PREFIX};${ENVDIR}" \
  -DCMAKE_INSTALL_PREFIX="${ENVDIR}" \
  -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=TRUE \
  -DCMAKE_INSTALL_RPATH="${ENVDIR}" \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
  -DFCL_INCLUDE_DIRS="${ENVDIR}/include/fcl" \
  -DPYTHON_EXECUTABLE="${CONDA_PREFIX}/bin/python" \
  -DPYTHON_INCLUDE_DIR="${CONDA_PREFIX}/include/python3.6m" \
  -DPYTHON_LIBRARY="${CONDA_PREFIX}/lib/libpython3.6m.so" \
  -Dpybind11_DIR="${CONDA_PREFIX}/share/cmake/pybind11" \
  ..

make -j$(nproc)
popd