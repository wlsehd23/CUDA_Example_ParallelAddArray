
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include <chrono> // To measure time with high precision

using namespace std;

void addWithCPU(int* c, int* a, int* b, unsigned int size);
cudaError_t addWithCuda(int *c, int *a, int *b, unsigned int size);
//CPU code for array sum
void addWithCPU(int* c, int* a, int* b, unsigned int size)
{
    for (int i = 0; i < size; i++) {
        c[i] = a[i] + b[i];
    }
}
// GPU code for array sum
__global__ void addKernel(int *c, int *a, int *b)
{
    int ti = blockIdx.x * blockDim.x + threadIdx.x;
    c[ti] = a[ti] + b[ti];
}

int main()
{
    unsigned int arraySize = 256*256;
    int* h_a = new int [arraySize];
    int* h_b = new int [arraySize];
    int* h_c = new int [arraySize];

    // insert values in array
    for (int i = 0; i < arraySize; i++) {
        h_a[i] = i;
        h_b[i] = 10 * i;
    }

    chrono::system_clock::time_point ct0, ct1;
    chrono::microseconds cDt1;

    // CPU
    ct0 = chrono::system_clock::now();
    addWithCPU(h_c, h_a, h_b, arraySize);
    ct1 = chrono::system_clock::now();
    cDt1 = chrono::duration_cast<chrono::microseconds>(ct1 - ct0);

    printf("CPU: {0,1,2,3,4,...} + {0,10,20,30,40,...} = {%d,%d,%d,%d,%d,...}\n",
        h_c[0], h_c[1], h_c[2], h_c[3], h_c[4]);
    printf("Elapsed time by CPU = %d us\n", cDt1.count());

    // GPU
    cudaError_t cudaStatus = addWithCuda(h_c, h_a, h_b, arraySize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    delete[] h_a, h_b, h_c;
    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *host_c, int * host_a, int * host_b, unsigned int size)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus=cudaSetDevice(0);

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus=cudaMalloc((void**)&dev_c, size * sizeof(int));
    cudaStatus=cudaMalloc((void**)&dev_a, size * sizeof(int));
    cudaStatus=cudaMalloc((void**)&dev_b, size * sizeof(int));

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, host_a, size * sizeof(int), cudaMemcpyHostToDevice);
    cudaStatus = cudaMemcpy(dev_b, host_b, size * sizeof(int), cudaMemcpyHostToDevice);

    // Launch a kernel on the GPU with one thread for each element.
    int BLOCK_SIZE = 256;
    int THREAD_SIZE = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;

    chrono::system_clock::time_point ct0, ct1;
    chrono::microseconds cDt1;
    ct0 = chrono::system_clock::now();
    addKernel<<<THREAD_SIZE, BLOCK_SIZE >>>(dev_c, dev_a, dev_b);
    ct1 = chrono::system_clock::now();
    cDt1 = chrono::duration_cast<chrono::microseconds>(ct1 - ct0);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(host_c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    
    // print result
    printf("GPU: {0,1,2,3,4,...} + {0,10,20,30,40,...} = {%d,%d,%d,%d,%d,...}\n",
        host_c[0], host_c[1], host_c[2], host_c[3], host_c[4]);
    printf("Elapsed time by GPU = %d us\n", cDt1.count());

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}
