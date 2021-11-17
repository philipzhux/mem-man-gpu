#include "virtual_memory.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
__device__ void user_program(VirtualMemory *vm, uchar *input, uchar *results,
                             int input_size,u32 pid) {
  for (int i = 0; i < input_size; i++){
    if(i%4==pid){
        vm_write(vm, i, input[i],pid);
    }
  }

  for (int i = input_size - 1; i >= input_size - 32769; i--)
    if(i%4==pid) int value = vm_read(vm, i,pid);
  
  for(u32 i = 0; i < input_size; i++) {
    if(i%4==pid) results[i] = vm_read(vm,i,pid);
  }

  // for(u32 i = input_sized/2; i < input_size; i++) {
  //   if(pid==1) results[i] = vm_read(vm,i,pid);
  // }
  //vm_snapshot(vm, results, 0, input_size,pid);
}
