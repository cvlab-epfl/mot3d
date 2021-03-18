rm -rdf build
mkdir build
cd build
cmake ..
make
cp *.so ..
cd ..
