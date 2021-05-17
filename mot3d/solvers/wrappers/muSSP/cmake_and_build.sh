rm -rdf build
mkdir build
cd build
cmake ..
exit_code=$?
if [ "$exit_code" -ne 0 ]; then
    >&2 echo "cmake failed with exit code ${exit_code}."
    exit $exit_code
fi
make
cp *.so ..
cd ..
