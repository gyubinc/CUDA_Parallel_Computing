#include <gputk.h>

#define gpuTKCheck(stmt)                                                     \
  do {                                                                       \
    cudaError_t err = stmt;                                                  \
    if (err != cudaSuccess) {                                                \
      gpuTKLog(ERROR, "Failed to run stmt ", #stmt);                         \
      return -1;                                                             \
    }                                                                        \
  } while (0)

#define Mask_width   5
#define Mask_radius  (Mask_width / 2)
#define TILE_WIDTH   16
#define w            (TILE_WIDTH + Mask_width - 1)
#define clamp(x)     (min(max((x), 0.0f), 1.0f))

__global__ void convolution(const float *inputImage,
                            const float *mask,
                            float *outputImage,
                            int channels,
                            int width,
                            int height,
                            int rowOffset) {
  //@@ INSERT CODE HERE
  //@@ NOTE: Use shared memory for operating convolution.
  // __shared__ float tile[w][w];

  //@@ TIP: Basic algorithm for convolution is as below.
  //@@ 1. Compute coordinates by using thread and block index.
  //@@ 2. Optimize code to use shared memory.

  //@@ Compute thread’s output pixel coords
  int row_o =
  int col_o =
  //@@ Check if current output pixel coords is inside image bounds
  if (row_o < height && col_o < width) {
      float sum = 0.0f;
      //@@ Iterate over the mask window
      for (int i = 0; i < Mask_width; i++) {
          for (int j = 0; j < Mask_width; j++) {
              //@@ Map mask element (i, j) to global input coords
              int in_r =
              int in_c =
              //@@ Check bounds before reading using in_r and in_c
              if (in_r >= 0 && in_r < height && in_c >= 0 && in_c < width) {
                  //@@ Calculate index and sum the calculated result
                  int inIdx =
                  int maskIdx =
                  sum += mask[maskIdx] * inputImage[inIdx]
              }
          }
      }
      //@@ Write clamped result back to the output image
      int outIdx = 
      outputImage[outIdx] = clamp(sum);
    }
}

int main(int argc, char *argv[]) {
  gpuTKArg_t arg;
  int maskRows;
  int maskColumns;
  int imageChannels;
  int imageWidth;
  int imageHeight;
  char *inputImageFile;
  char *inputMaskFile;
  gpuTKImage_t inputImage;
  gpuTKImage_t outputImage;
  float *hostInputImageData;
  float *hostOutputImageData;
  float *hostMaskData;
  float *deviceInputImageData;
  float *deviceOutputImageData;
  float *deviceMaskData;

  //@@ NOTE: Use streams and pinned memory.
  const int numStreams = 4;
  cudaStream_t streams[numStreams];
  float *pinnedInput = nullptr;
  float *pinnedOutput = nullptr;

  arg           = gpuTKArg_read(argc, argv);

  inputImageFile = gpuTKArg_getInputFile(arg, 0);
  inputMaskFile  = gpuTKArg_getInputFile(arg, 1);

  inputImage     = gpuTKImport(inputImageFile);
  hostMaskData   = (float *)gpuTKImport(inputMaskFile, &maskRows, &maskColumns);

  assert(maskRows == Mask_width);
  assert(maskColumns == Mask_width);

  imageWidth      = gpuTKImage_getWidth(inputImage);
  imageHeight     = gpuTKImage_getHeight(inputImage);
  imageChannels   = gpuTKImage_getChannels(inputImage);

  outputImage     = gpuTKImage_new(imageWidth, imageHeight, imageChannels);

  hostInputImageData  = gpuTKImage_getData(inputImage);
  hostOutputImageData = gpuTKImage_getData(outputImage);

  gpuTKTime_start(GPU, "Doing GPU Computation (memory + compute)");

  gpuTKTime_start(GPU, "Doing GPU memory allocation");
  //@@ INSERT CODE HERE
  //@@ NOTE: use pinned memory and streams.
  gpuTKTime_stop(GPU, "Doing GPU memory allocation");

  gpuTKTime_start(Copy, "Copying data to the GPU");
  //@@ INSERT CODE HERE
  gpuTKTime_stop(Copy, "Copying data to the GPU");

  gpuTKTime_start(Compute, "Doing the computation on the GPU");
  //@@ INSERT CODE HERE
  //@@ NOTE: use below code to call convolution kernel.
  //  convolution<<<dimGrid, dimBlock>>>(deviceInputImageData, deviceMaskData,
  //                                     deviceOutputImageData, imageChannels,
  //                                     imageWidth, imageHeight);
  gpuTKTime_stop(Compute, "Doing the computation on the GPU");

  gpuTKTime_start(Copy, "Copying data from the GPU");
  //@@ INSERT CODE HERE
  gpuTKTime_stop(Copy, "Copying data from the GPU");

  gpuTKTime_stop(GPU, "Doing GPU Computation (memory + compute)");

  gpuTKSolution(arg, outputImage);

  //@@ Insert code here

  free(hostMaskData);
  gpuTKImage_delete(outputImage);
  gpuTKImage_delete(inputImage);

  return 0;
}