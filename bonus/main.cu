#include "virtual_memory.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>

#define DATAFILE "./data.bin"
#define OUTFILE "./snapshot.bin"

// page size is 32bytes
#define PAGE_SIZE_C (1 << 5)
// 16 KB in page table
#define INVERT_PAGE_TABLE_SIZE_C (1 << 14)
// 32 KB in shared memory
#define PHYSICAL_MEM_SIZE_C (1 << 15)
// 128 KB in global memory
#define STORAGE_SIZE_C (1 << 17)

//// count the pagefault times
__device__ __managed__ int pagefault_num = 0;

// data input and output
__device__ __managed__ uchar results[STORAGE_SIZE_C];
__device__ __managed__ uchar input[STORAGE_SIZE_C];

// memory allocation for virtual_memory
// secondary memory
__device__ __managed__ uchar storage[STORAGE_SIZE_C];
// page table
extern __shared__ u32 pt[];






/* Lock not applicable in the same block
   Putting threads in different blocks is
   not applicable either as teh shared memory
   can't be shared across blocks, therefore
   the lock method is abandoned
*/

// Lock::Lock(void){
//     int state[5];
//     for(int i=0; i<5; i++){
//       state[i] = 0;
//     }
//     cudaMalloc((void**) &mutex, sizeof(int)*5);
//     cudaMemcpy(mutex, &state, sizeof(int)*5, cudaMemcpyHostToDevice);
// }

// Lock::~Lock(void){
//     cudaFree(mutex);
// }

// __device__ void Lock::lock(int index){
//     while(atomicCAS(mutex+index, 0, 1) != 0);
//   }

// __device__ void Lock::unlock(int index){
//     atomicExch(mutex+index, 0);
// }


// __device__ void Lock::try_lock(int index){
//     atomicExch(mutex+index, 1);
// }
__global__ void mykernel(int input_size) {

  // memory allocation for virtual_memory
  // take shared memory as physical memory
  __shared__ uchar data[PHYSICAL_MEM_SIZE_C];
  // __shared__ u32 pt[(INVERT_PAGE_TABLE_SIZE_C>>2)];
  u32 index = threadIdx.x;
  VirtualMemory vm;
  vm.buffer = data;
  vm.storage = storage;
  vm.invert_page_table = pt;
  vm.pagefault_num_ptr = &pagefault_num;

  // init constants
  vm.PAGESIZE = PAGE_SIZE_C;
  vm.INVERT_PAGE_TABLE_SIZE = INVERT_PAGE_TABLE_SIZE_C;
  vm.PHYSICAL_MEM_SIZE = PHYSICAL_MEM_SIZE_C;
  vm.STORAGE_SIZE = STORAGE_SIZE_C;
  vm.PAGE_ENTRIES = PHYSICAL_MEM_SIZE_C / PAGE_SIZE_C;
  init_invert_page_table(&vm);
  __syncthreads();
  // user program the access pattern for testing paging
  if(index==0) user_program(&vm, input, results, input_size, index);
  __syncthreads();
  if(index==1) user_program(&vm, input, results, input_size, index);
  __syncthreads();
  if(index==2) user_program(&vm, input, results, input_size, index);
  __syncthreads();
  if(index==3) user_program(&vm, input, results, input_size, index);

}

__host__ void write_binaryFile(char *fileName, void *buffer, int bufferSize) {
  FILE *fp;
  fp = fopen(fileName, "wb");
  fwrite(buffer, 1, bufferSize, fp);
  fclose(fp);
}

__host__ int load_binaryFile(char *fileName, void *buffer, int bufferSize) {
  FILE *fp;

  fp = fopen(fileName, "rb");
  if (!fp) {
    printf("***Unable to open file %s***\n", fileName);
    exit(1);
  }

  // Get file length
  fseek(fp, 0, SEEK_END);
  int fileLen = ftell(fp);
  fseek(fp, 0, SEEK_SET);

  if (fileLen > bufferSize) {
    printf("****invalid testcase!!****\n");
    printf("****software warrning: the file: %s size****\n", fileName);
    printf("****is greater than buffer size****\n");
    exit(1);
  }

  // Read file contents into buffer
  fread(buffer, fileLen, 1, fp);
  fclose(fp);

  return fileLen;
}

int main() {
  cudaError_t cudaStatus;
  int input_size = load_binaryFile(DATAFILE, input, STORAGE_SIZE_C);

  /* Launch kernel function in GPU, with single thread
  and dynamically allocate INVERT_PAGE_TABLE_SIZE bytes of share memory,
  which is used for variables declared as "extern __shared__" */
  // Lock lock_set;
  mykernel<<<1, 4, INVERT_PAGE_TABLE_SIZE_C>>>(input_size);

  cudaStatus = cudaGetLastError();
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "mykernel launch failed: %s\n",
            cudaGetErrorString(cudaStatus));
    return 0;
  }

  printf("input size: %d\n", input_size);

  cudaDeviceSynchronize();
  cudaDeviceReset();

  write_binaryFile(OUTFILE, results, input_size);

  printf("pagefault number is %d\n", pagefault_num);

  return 0;
}
