// CUDA: grid stride looping
#define CUDA_KERNEL_LOOP(i, n)                        \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
       i < (n);                                       \
       i += blockDim.x * gridDim.x)

__global__ void MaxPoolForward(const int nthreads, const float* input_data,
    const int channels, const int length, const int height, const int width,
          const int pooled_length, const int pooled_height, const int pooled_width,
    const int kernel_height, const int kernel_width, const int kernel_depth,
    const int stride_height, const int stride_width, const int temporal_stride,
                               float* output_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int pl = (index / pooled_width / pooled_height) % pooled_length;
    int c = (index / pooled_width / pooled_height / pooled_length) % channels;
    int n = index / pooled_width / pooled_height / pooled_length / channels;
    int hstart = ph * stride_height;
    int hend = min(hstart + kernel_height, height);
    int wstart = pw * stride_width;
    int wend = min(wstart + kernel_width, width);
    int lstart = pl * temporal_stride;
    int lend = min(lstart + kernel_depth, length);
    float maxval = -FLT_MAX;
    input_data += (n * channels + c) * length * height * width;
    for (int l = lstart; l < lend; ++l) {
      for (int h = hstart; h < hend; ++h) {
        for (int w = wstart; w < wend; ++w) {
          maxval = max(maxval, input_data[(l * height + h) * width + w]);
        }
      }
    }
    output_data[index] = maxval;

  }
}

__global__ void MaxPoolBackward(const int nthreads, const float* input_data,
    const float* output_data, const float* output_diff,
    const int channels, const int length, const int height,
    const int width, const int pooled_length, const int pooled_height, const int pooled_width,
    const int kernel_size, const int kernel_depth, const int stride, const int temporal_stride, float* input_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // find out the local index
    // find out the local offset
    int w = index % width;
    int h = (index / width) % height;
    int l = (index / width / height) % length;
    int c = (index / width / height / length) % channels;
    int n = index / width / height / length / channels;

    int phstart = (h < kernel_size) ? 0 : (h - kernel_size) / stride + 1;
    int phend = min(h / stride + 1, pooled_height);
    int pwstart = (w < kernel_size) ? 0 : (w - kernel_size) / stride + 1;
    int pwend = min(w / stride + 1, pooled_width);
    int plstart = (l < kernel_depth) ? 0 : (l - kernel_depth) / temporal_stride + 1;
    int plend = min(l / temporal_stride + 1, pooled_length);

    float gradient = 0;
    float input_datum =
        input_data[(((n * channels + c) * length + l) * height + h) * width + w];
    output_data += (n * channels + c) * pooled_length * pooled_height * pooled_width;
    output_diff += (n * channels + c) * pooled_length * pooled_height * pooled_width;
    for (int pl = plstart; pl < plend; ++pl) {
      for (int ph = phstart; ph < phend; ++ph) {
        for (int pw = pwstart; pw < pwend; ++pw) {
          gradient += output_diff[(pl * pooled_height + ph) * pooled_width + pw] *
              (input_datum == output_data[(pl * pooled_height + ph) * pooled_width + pw]);
        }
      }
    }
    input_diff[index] = gradient;
  }
}


static int cunn_VolumetricMaxPooling_updateOutput(lua_State *L) {
  // Input
  THCudaTensor *input = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
  // Params:
  int dD = luaT_getfieldcheckint(L, 1, "dW");
  int dW = luaT_getfieldcheckint(L, 1, "dH");
  int dH = luaT_getfieldcheckint(L, 1, "dT");
  int kD = luaT_getfieldcheckint(L, 1, "kW");
  int kW = luaT_getfieldcheckint(L, 1, "kH");
  int kH = luaT_getfieldcheckint(L, 1, "kT");

  THCudaTensor *output = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "output", "torch.CudaTensor");

  const int device = THCudaTensor_getDevice(input);
  luaL_argcheck(L, THCudaTensor_getDevice(output) == device ||
                THCudaTensor_getDevice(output) == -1, 1,
                "input and output need to be on the same device");

  luaL_argcheck(L, input->nDimension == 4 || input->nDimension == 5, 2, "4D or 5D (batch mode) tensor is expected");

  int batch = 1;
  if (input->nDimension == 4) {
    // Force batch
    batch = 0;
    THCudaTensor_resize5d(input, 1, input->size[0], input->size[1],
                          input->size[2], input->size[3]);
  }

  int nInputPlane = input->size[1];
  long inputWidth   = input->size[3];
  long inputHeight  = input->size[2];
  long inputDepth   = input->size[4];
  long outputWidth  = (inputWidth  - kW) / dW + 1;
  long outputHeight = (inputHeight - kH) / dH + 1;
  long outputDepth  = (inputDepth - kD) / dD + 1;

  // Batch size + input planes
  long batchSize = input->size[0];

  // Resize output
  THCudaTensor_resize5d(output, batchSize, nInputPlane, outputDepth,
                        outputHeight, outputWidth);

  float* input_data = THCudaTensor_data(input);
  float* output_data = THCudaTensor_data(output);
  const int nBlocks = (nInputPlane + 1024 - 1) / 1024;
  MaxPoolForward<<<nBlocks, 1024>>>(
    nInputPlane, input_data, nInputPlane, inputDepth,
    inputHeight, inputWidth, outputDepth, outputHeight, outputWidth,
    kH, kW, kD, dH, dW, dD, output_data);

  // Resize output
  if (batch == 0) {
    THCudaTensor_resize4d(output, nInputPlane, outputHeight, outputWidth, outputDepth);
    THCudaTensor_resize4d(input, nInputPlane, inputHeight, inputWidth, inputDepth);
  }

  // return output
  return 1;
}

static int cunn_VolumetricMaxPooling_updateGradInput(lua_State *L) {
  // Inputs
  THCudaTensor *input = (THCudaTensor *)luaT_checkudata(L, 2, "torch.CudaTensor");
  THCudaTensor *gradOutput = (THCudaTensor *)luaT_checkudata(L, 3, "torch.CudaTensor");

  // Params
  int dD = luaT_getfieldcheckint(L, 1, "dW");
  int dW = luaT_getfieldcheckint(L, 1, "dH");
  int dH = luaT_getfieldcheckint(L, 1, "dT");
  int kD = luaT_getfieldcheckint(L, 1, "kW");
  int kW = luaT_getfieldcheckint(L, 1, "kH");
  int kH = luaT_getfieldcheckint(L, 1, "kT");

  THCudaTensor *weight = (THCudaTensor *)luaT_getfieldcheckudata(L, 1, "weight", "torch.CudaTensor");
  THCudaTensor *gradColumns = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "finput", "torch.CudaTensor");
  THCudaTensor *gradInput = (THCudaTensor *)luaT_getfieldcheckudata(L, 1, "gradInput", "torch.CudaTensor");

  luaL_argcheck(L, input->nDimension == 4 || input->nDimension == 5, 2, "4D or 5D (batch mode) tensor is expected");

  int batch = 1;
  if (input->nDimension == 4) {
    // Force batch
    batch = 0;
    THCudaTensor_resize5d(input, 1, input->size[0], input->size[1], input->size[2], input->size[3]);
    THCudaTensor_resize5d(gradOutput, 1, gradOutput->size[0], gradOutput->size[1], gradOutput->size[2], gradOutput->size[3]);
  }

  long nInputPlane  = input->size[3];
  long inputWidth   = input->size[3];
  long inputHeight  = input->size[2];
  long inputDepth   = input->size[4];
  long outputWidth  = (inputWidth - kW) / dW + 1;
  long outputHeight = (inputHeight - kH) / dH + 1;
  long outputDepth  = (inputDepth - kD) / dD + 1;

  // Batch size + input planes
  long batchSize = input->size[0];

  // Resize output
  THCudaTensor_resize5d(gradInput, batchSize, nInputPlane, inputDepth, inputHeight, inputWidth);

  // Resize output
  if (batch == 0) {
    THCudaTensor_resize4d(gradOutput, nInputPlane, outputHeight, outputWidth, outputDepth);
    THCudaTensor_resize4d(input, nInputPlane, inputHeight, inputWidth, inputDepth);
    THCudaTensor_resize4d(gradInput, nInputPlane, inputHeight, inputWidth, inputDepth);
  }

  // Return gradInput
  return 1;
}

static const struct luaL_Reg cunn_VolumetricMaxPooling__ [] = {
  {"VolumetricMaxPooling_updateOutput", cunn_VolumetricMaxPooling_updateOutput},
  {"VolumetricMaxPooling_updateGradInput", cunn_VolumetricMaxPooling_updateGradInput},
  {NULL, NULL}
};

static void cunn_VolumetricMaxPooling_init(lua_State *L)
{
  luaT_pushmetatable(L, "torch.CudaTensor");
  luaT_registeratname(L, cunn_VolumetricMaxPooling__, "nn");
  lua_pop(L,1);
}

#undef CUDA_KERNEL_LOOP
