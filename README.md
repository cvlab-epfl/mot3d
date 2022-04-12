# Fast Single and Multi Camera Multi-object Tracking using Minimum Cost Maximum Flow Algorithm

This package provides all the necessary routines to create and solve minimum cost maximum flow graphs for 2D or 3D multi object tracking.
In the projects folder we provide solutions for single view and multiple view online multi-object tracking for the PETS2009 sequence.

#### Solvers:
- Integer Linear Programming (ILP) [Slow]
- Minimum-update Successive Shortest Path (muSSP) [Very Fast] https://github.com/yu-lab-vt/muSSP

Big thanks to authors of muSSP solver for releasing the code. Please cite their paper if you find it useful:
```
@inproceedings{wang2019mussp,
  title={muSSP: Efficient Min-cost Flow Algorithm for Multi-object Tracking},
  author={Wang, Congchao and Wang, Yizhi and Wang, Yinxue and Wu, Chiung-Ting and Yu, Guoqiang},
  booktitle={Advances in Neural Information Processing Systems},
  pages={423--432},
  year={2019}
}
```

## Installation
```
export PYTHONPATH="...parent folder.../mot3d:$PYTHONPATH"
```
#### Build muSSP

##### Linux
```
sudo apt-get update
sudo apt-get install unzip cmake build-essential libboost-python-dev
cd ..../mot3d
pip install -r requirements.txt
unzip mussp-master-6cf61b8.zip
cd ..../mot3d/mot3d/solvers/wrappers/muSSP
./cmake_and_build.sh
```
##### Windows
```
cd ..../mot3d
pip install -r requirements.txt
```
decompress mussp-master-6cf61b8.zip in the same location. The folder must be named `muSSP-master` and containing `muSSP`, `SSP`, `FolowMe` and so on.

###### Installing Boost.Python for Windows
Download boost https://www.boost.org/users/download/
unzip it then run `.\bootstrap.bat` from command line.
If you have an error such as `'"cl"' is not recognized as an internal or external command` you probably have to install Visual Studio https://visualstudio.microsoft.com/it/downloads/. Make sure to install MSVC compiler and Windows SDK by ticking on Individual Components->Compilers/build-tools->MSVC 143 C++ x86_x64 (Latest) and Windows SDK 11
If there is another error about missing header files you are probably missing Windows SDK.
```
.\b2 -j12 --with-python --prefix=C:\Boost --libdir=C:\Boost\lib --includedir=C:\Boost\include install
```
if the installation went well, you should have this library `C:\Boost\lib\libboost_python3*.lib` and this folder `C:\Boost\include`
###### Compiling muSSP C++ wrapper code
```
cd ..../mot3d/mot3d/solvers/wrappers/muSSP
rmdir /s build      :: Always a good idea to delete/clean the build folder if it exist already
mkdir build
cd build
"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
cmake -DBOOST_ROOT=C:\\Boost -DBOOST_PYTHON_STATIC_LIB=ON ..
```
At this point, if no errors, open the file `wrapper.sln` with Visual Studio, then set Release mode then press Build Solution.
If no errors, you should see `========== Build: 2 succeeded, 0 failed`. If you have an error such as `Missing dependency Graph.h file` you have a problem with the unzipping of the mussp-master.zip. Once fixed you have to re-do cmake command as well.

At this point you should have this file `build/Release/wrapper.pyd`. Copy it in the parent folder of `build` where there is the file `wrapper.cpp`.
```
copy ./release/wrapper.pyd ..
```
Installation done!

## Usage
Check the examples!
