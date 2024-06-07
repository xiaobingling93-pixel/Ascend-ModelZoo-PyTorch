rm -r build
mkdir build
cd build

TorchPath="torch/path/you/should/set"

cmake .. -DTORCH_ROOT=${TorchPath} -DCMAKE_PREFIX_PATH=${TorchPath}
make -j