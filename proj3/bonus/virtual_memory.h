#ifndef VIRTUAL_MEMORY_H
#define VIRTUAL_MEMORY_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <inttypes.h>

#define DATA_LOCK 4

#define VALID_BIT 31
#define DIRTY_BIT 30


#define GET_BIT(b,t) ((t & (0x1<<b))>>b)
#define SET_BIT(b,t) t = t | (0x1<<b)
#define UNSET_BIT(b,t) t = t & ~(0x1<<b)


#define GET_PID(vm,index) ((vm->invert_page_table[index*2]>>13) & 0x3)
#define GET_ADDR(vm,index) (vm->invert_page_table[index*2] & 0x1FFF)
#define GET_NEXT(vm,index) (vm->invert_page_table[index*2+1] & 0x3FF)
#define GET_PREV(vm,index) ((vm->invert_page_table[index*2+1]>>11) & 0x3FF)


#define IS_EMPTY(vm) GET_BIT(22,vm->invert_page_table[5])
#define SET_EMPTY(vm) SET_BIT(22,vm->invert_page_table[5])
#define UNSET_EMPTY(vm) UNSET_BIT(22,vm->invert_page_table[5])


#define GET_HEAD_NODE(vm) ((vm->invert_page_table[1]>>22) & 0x3FF)
#define GET_TAIL_NODE(vm) ((vm->invert_page_table[3]>>22) & 0x3FF)
#define SET_HEAD_NODE(vm,head) vm->invert_page_table[1] = ((vm->invert_page_table[1] & ~(0x3FF<<22))|((head&0x3FF)<<22))
#define SET_TAIL_NODE(vm,tail) vm->invert_page_table[3] = ((vm->invert_page_table[3] & ~(0x3FF<<22))|((tail&0x3FF)<<22))


#define IS_DIRTY(vm,index) GET_BIT(VALID_BIT,vm->invert_page_table[index*2])
#define IS_INVALID(vm,index) GET_BIT(DIRTY_BIT,vm->invert_page_table[index*2])
#define SET_DIRTY(vm,index) SET_BIT(VALID_BIT,vm->invert_page_table[index*2])
#define SET_INVALID(vm,index) SET_BIT(DIRTY_BIT,vm->invert_page_table[index*2])
#define UNSET_DIRTY(vm,index) UNSET_BIT(VALID_BIT,vm->invert_page_table[index*2])
#define UNSET_INVALID(vm,index) UNSET_BIT(DIRTY_BIT,vm->invert_page_table[index*2])


#define SET_PID (vm,index,pid) vm->invert_page_table[index*2] = (vm->invert_page_table[index*2] & (~(0x3<<13)) | ((pid & 0x3) << 13))
#define SET_ADDR(vm,index,addr,pid) vm->invert_page_table[index*2] = (vm->invert_page_table[index*2]& ~(0x1FFF))|(addr & 0x1FFF); SET_PID (vm,index,pid)


#define SET_NEXT(vm,index,next_index) vm->invert_page_table[index*2+1] =\
(vm->invert_page_table[index*2+1] & ~(0x3FF))|(next_index & 0x3FF); UNSET_NEXT_NULL(vm,index);
#define SET_PREV(vm,index,prev_index) vm->invert_page_table[index*2+1] =\
(vm->invert_page_table[index*2+1] & ~(0x3FF<<11))| ((prev_index & 0x3FF)<<11); UNSET_PREV_NULL(vm,index);


#define SET_NEXT_NULL(vm,index) SET_BIT(10,vm->invert_page_table[index*2+1])
#define SET_PREV_NULL(vm,index) SET_BIT(21,vm->invert_page_table[index*2+1])
#define UNSET_NEXT_NULL(vm,index) UNSET_BIT(10,vm->invert_page_table[index*2+1])
#define UNSET_PREV_NULL(vm,index) UNSET_BIT(21,vm->invert_page_table[index*2+1])
#define NEXT_IS_NULL(vm,index) GET_BIT(10,vm->invert_page_table[index*2+1])
#define PREV_IS_NULL(vm,index) GET_BIT(21,vm->invert_page_table[index*2+1])


#define INIT(vm,index) vm->invert_page_table[index*2] = 0; vm->invert_page_table[index*2+1] = 0
#define INIT_DISK_MAP(vm,index) vm->invert_page_table[vm->PAGE_ENTRIES*2+index/2] = \
(vm->invert_page_table[vm->PAGE_ENTRIES*2+index/2] & ~(0xFFFF<<((index%2)*16))) | ((((1<<15))) << ((index%2)*16))


#define GET_DISK_IMAP(vm,index) ((vm->invert_page_table[vm->PAGE_ENTRIES*2+index/2]) & (0xFFFF<<((index%2)*16))) >> ((index%2)*16)
#define GET_VM_FROM_DISK(vm,index) (GET_DISK_IMAP(vm,index) & 0x1FFF)
#define GET_PID_FROM_DISK(vm,index) ((GET_DISK_IMAP(vm,index) & 0x3<<13)>>13)
#define DISK_IS_INVALID(vm,index) GET_BIT(15,GET_DISK_IMAP(vm,index))
#define SET_DISK_INVALID(vm,index) SET_BIT(15+(index%2)*16,vm->invert_page_table[vm->PAGE_ENTRIES*2+index/2])
#define UNSET_DISK_INVALID(vm,index) UNSET_BIT(15+(index%2)*16,vm->invert_page_table[vm->PAGE_ENTRIES*2+index/2])
#define CONSTRUCT_DISK_IMAP(pid,vm_addr) ((((vm_addr & 0x1FFF) | ((pid & 0x3) <<13))) & ~(1<<15))
#define SET_DISK_TO_VM(vm,index,pid,vm_addr) vm->invert_page_table[vm->PAGE_ENTRIES*2+index/2] = \
(vm->invert_page_table[vm->PAGE_ENTRIES*2+index/2] & ~(0xFFFF<<((index%2)*16)) | (CONSTRUCT_DISK_IMAP(pid,vm_addr) << ((index%2)*16)))


#define SET_COUNT(vm,count) (vm->invert_page_table[(vm->PAGE_ENTRIES-1)*2] = ((vm->invert_page_table[(vm->PAGE_ENTRIES-1)*2] & ~(0x7FF<<15)) | (count<<15)))
#define GET_COUNT(vm) ((vm->invert_page_table[(vm->PAGE_ENTRIES-1)*2] & (0x7FF<<15)) >> 15) //11 bit count


#define HASH(key,i) (key%1021+i)%1024
#define HASH_DISK(key,i) (key%4099+i)%4096


typedef unsigned char uchar;
typedef uint32_t u32;

struct VirtualMemory {
  uchar *buffer;
  uchar *storage;
  u32 *invert_page_table;
  int *pagefault_num_ptr;

  int PAGESIZE;
  int INVERT_PAGE_TABLE_SIZE;
  int PHYSICAL_MEM_SIZE;
  int STORAGE_SIZE;
  int PAGE_ENTRIES;
};

struct Lock{
  int *mutex;
  Lock(void);
  ~Lock(void);
  __device__ void lock();
  __device__ void unlock();
};

// prototypes
__device__ void vm_init(VirtualMemory *vm, uchar *buffer, uchar *storage,
                        u32 *invert_page_table, int *pagefault_num_ptr,
                        int PAGESIZE, int INVERT_PAGE_TABLE_SIZE,
                        int PHYSICAL_MEM_SIZE, int STORAGE_SIZE,
                        int PAGE_ENTRIES);
__device__ uchar vm_read(VirtualMemory *vm, u32 addr);
__device__ void vm_write(VirtualMemory *vm, u32 addr, uchar value);
__device__ void vm_snapshot(VirtualMemory *vm, uchar *results, int offset,
                            int input_size);
__device__ void mark_use(VirtualMemory *vm, u32 mem_entry);
__device__ void swap_in(VirtualMemory *vm,u32 mem_entry, u32 disk_entry);
__device__ u32 extract_lru(VirtualMemory *vm);
__device__ u32 evict_lru(VirtualMemory *vm);
__device__ void swap_out(VirtualMemory *vm,u32 mem_entry, u32 disk_entry, u32 vm_addr);

#endif
