all: ;nvcc *.cu --relocatable-device-code=true -I. -o vm
clean: ;rm vm snapshot.bin