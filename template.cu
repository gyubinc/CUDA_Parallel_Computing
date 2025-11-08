#include <gputk.h>

#define gpuTKCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      gpuTKLog(ERROR, "Failed to run stmt ", #stmt);                         \
      gpuTKLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

// Compute C = A * B
#ifndef TILE_WIDTH
#define TILE_WIDTH 128
#endif

#ifndef SHARED_MEM_CARVEOUT_PERCENT
#define SHARED_MEM_CARVEOUT_PERCENT 0
#endif

__global__ void matrixMultiplyShared(float *A, float *B, float *C,
                                     int numARows, int numAColumns,
                                     int numBRows, int numBColumns,
                                     int numCRows, int numCColumns) {
  //@@ Insert code to implement matrix multiplication here
  //@@ You have to use shared memory for this lab
  __shared__ float ds_A[TILE_WIDTH][TILE_WIDTH];
  __shared__ float ds_B[TILE_WIDTH][TILE_WIDTH];

  int Row = blockIdx.y * blockDim.y + threadIdx.y;
  int Col = blockIdx.x * blockDim.x + threadIdx.x;

  float pvalue = 0.0f;
  int numTiles = (numAColumns + TILE_WIDTH - 1) / TILE_WIDTH;

  for (int tileIdx = 0; tileIdx < numTiles; ++tileIdx) {
    int ACol = tileIdx * TILE_WIDTH + threadIdx.x;
    int ARow = Row;
    if (ARow < numARows && ACol < numAColumns) {
      ds_A[threadIdx.y][threadIdx.x] =
          A[ARow * numAColumns + ACol];
    } else {
      ds_A[threadIdx.y][threadIdx.x] = 0.0f;
    }

    int BRow = tileIdx * TILE_WIDTH + threadIdx.y;
    int BCol = Col;
    if (BRow < numBRows && BCol < numBColumns) {
      ds_B[threadIdx.y][threadIdx.x] =
          B[BRow * numBColumns + BCol];
    } else {
      ds_B[threadIdx.y][threadIdx.x] = 0.0f;
    }

    __syncthreads();

    for (int k = 0; k < TILE_WIDTH; ++k) {
      pvalue += ds_A[threadIdx.y][k] * ds_B[k][threadIdx.x];
    }

    __syncthreads();
  }

  if (Row < numCRows && Col < numCColumns) {
    C[Row * numCColumns + Col] = pvalue;
  }
}

int main(int argc, char **argv) {
  gpuTKArg_t args;
  float *hostA; // The A matrix
  float *hostB; // The B matrix
  float *hostC; // The output C matrix
  float *deviceA;
  float *deviceB;
  float *deviceC;
  int numARows;    // number of rows in the matrix A
  int numAColumns; // number of columns in the matrix A
  int numBRows;    // number of rows in the matrix B
  int numBColumns; // number of columns in the matrix B
  int numCRows;    // number of rows in the matrix C (you have to set this)
  int numCColumns; // number of columns in the matrix C (you have to set
                   // this)

  args = gpuTKArg_read(argc, argv);

  gpuTKTime_start(Generic, "Importing data and creating memory on host");
  hostA = (float *)gpuTKImport(gpuTKArg_getInputFile(args, 0), &numARows,
                            &numAColumns);
  hostB = (float *)gpuTKImport(gpuTKArg_getInputFile(args, 1), &numBRows,
                            &numBColumns);
  //@@ Set numCRows and numCColumns
  numCRows    = numARows;
  numCColumns = numBColumns;
  //@@ Allocate the hostC matrix
  size_t matrixSizeC = numCRows;
  matrixSizeC *= numCColumns;
  hostC = (float *)malloc(matrixSizeC * sizeof(float));

  gpuTKTime_stop(Generic, "Importing data and creating memory on host");

  gpuTKLog(TRACE, "The dimensions of A are ", numARows, " x ", numAColumns);
  gpuTKLog(TRACE, "The dimensions of B are ", numBRows, " x ", numBColumns);

  gpuTKTime_start(GPU, "Allocating GPU memory.");
  //@@ Allocate GPU memory here
  gpuTKCheck(cudaMalloc((void **)&deviceA,
                        sizeof(float) * numARows * numAColumns));
  gpuTKCheck(cudaMalloc((void **)&deviceB,
                        sizeof(float) * numBRows * numBColumns));
  gpuTKCheck(cudaMalloc((void **)&deviceC,
                        sizeof(float) * numCRows * numCColumns));

  gpuTKTime_stop(GPU, "Allocating GPU memory.");

  gpuTKTime_start(GPU, "Copying input memory to the GPU.");
  //@@ Copy memory to the GPU here
  gpuTKCheck(cudaMemcpy(deviceA, hostA,
                        sizeof(float) * numARows * numAColumns,
                        cudaMemcpyHostToDevice));
  gpuTKCheck(cudaMemcpy(deviceB, hostB,
                        sizeof(float) * numBRows * numBColumns,
                        cudaMemcpyHostToDevice));

  gpuTKTime_stop(GPU, "Copying input memory to the GPU.");

  //@@ Initialize the grid and block dimensions here
  dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
  dim3 dimGrid((numCColumns + TILE_WIDTH - 1) / TILE_WIDTH,
               (numCRows + TILE_WIDTH - 1) / TILE_WIDTH, 1);

  const int preferredSharedMemCarveout = SHARED_MEM_CARVEOUT_PERCENT;
  gpuTKLog(TRACE, "Preferred shared memory carveout is ",
           preferredSharedMemCarveout, "%");
  gpuTKCheck(cudaFuncSetAttribute((const void *)matrixMultiplyShared,
                                  cudaFuncAttributePreferredSharedMemoryCarveout,
                                  preferredSharedMemCarveout));

  gpuTKTime_start(Compute, "Performing CUDA computation");
  //@@ Launch the GPU Kernel here
  matrixMultiplyShared<<<dimGrid, dimBlock>>>(deviceA, deviceB, deviceC,
                                              numARows, numAColumns,
                                              numBRows, numBColumns,
                                              numCRows, numCColumns);
  gpuTKCheck(cudaGetLastError());

  cudaDeviceSynchronize();
  gpuTKTime_stop(Compute, "Performing CUDA computation");

  gpuTKTime_start(Copy, "Copying output memory to the CPU");
  //@@ Copy the GPU memory back to the CPU here
  gpuTKCheck(cudaMemcpy(hostC, deviceC,
                        sizeof(float) * numCRows * numCColumns,
                        cudaMemcpyDeviceToHost));

  gpuTKTime_stop(Copy, "Copying output memory to the CPU");

  gpuTKTime_start(GPU, "Freeing GPU Memory");
  //@@ Free the GPU memory here
  gpuTKCheck(cudaFree(deviceA));
  gpuTKCheck(cudaFree(deviceB));
  gpuTKCheck(cudaFree(deviceC));

  gpuTKTime_stop(GPU, "Freeing GPU Memory");

  gpuTKSolution(args, hostC, numCRows, numCColumns);

  free(hostA);
  free(hostB);
  free(hostC);

  return 0;
}
