#include <cuda.h>
#include <cuda_runtime.h>

#include <torch/extension.h>

namespace decode_captions {


__device__ int str_len(const char* string) {
    int length = 0;

    while (string[length] != '\0') {
        length++;
    }

    return length;
}


__global__ void decode_kernel(const int64_t* encoded_captions, const char** idx2key, char* decoded_tokens, int num_captions, int caption_length, int buffer_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Return if idx is out of bounds for encoded captions
    if (idx >= (num_captions * caption_length)) return;

    int caption_idx = idx / num_captions;
    int token_idx = idx % caption_length;

    if (token_idx >= caption_length) return;

    int token_id = encoded_captions[caption_idx * caption_length + token_idx];

    const char* token = idx2key[token_id];
    const int token_len = str_len(token);

    char* buffer = &decoded_tokens[caption_idx * buffer_size + token_idx];
    for (int i = 0; i < token_len; i++) {
        buffer[i] = token[i];
    }

}


std::vector<std::string> decodeBatchedCaptions_cuda(const at::Tensor& encoded_captions, const std::vector<std::string>& idx2key, const int64_t& max_token_length) {
    // Run checks to make sure tensor has correct dtype and is on correct device
    TORCH_CHECK(encoded_captions.dtype() == at::kLong, "Input tensor must have dtype Long");
    TORCH_CHECK(encoded_captions.device().type() == at::DeviceType::CUDA, "Input tensor must be on GPU");

    std::cout << "CUDA" << std::endl;

    // Initialize number of captions, length of each caption, and size of result buffer
    const int num_captions = encoded_captions.size(0);
    const int caption_length = encoded_captions.size(1);
    const int buffer_size = caption_length * max_token_length;

    // Create pointer for encoded captions
    at::Tensor encoded_captions_contig = encoded_captions.contiguous();
    const int64_t* encoded_captions_ptr = encoded_captions_contig.data_ptr<int64_t>();

    // Create result buffer for decoded tokens
    at::Tensor decoded_tokens = at::empty({num_captions, buffer_size}, at::kChar);
    char* decoded_tokens_ptr;

    cudaMalloc(&decoded_tokens_ptr, num_captions * buffer_size * sizeof(char));

    // Create pointer to idx2key and copy data to GPU
    const char** idx2key_ptr;
    cudaMalloc(&idx2key_ptr, idx2key.size() * sizeof(const char*));
    cudaMemcpy(idx2key_ptr, idx2key.data(), idx2key.size() * sizeof(const char*), cudaMemcpyHostToDevice);

    // Initialize block size and number of blocks to use for CUDA kernel
    const int blockSize = 256;
    const int numBlocks = ((num_captions * caption_length) + blockSize - 1) / blockSize;

    decode_kernel<<<numBlocks, blockSize>>>(encoded_captions_ptr, idx2key_ptr, decoded_tokens_ptr, num_captions, caption_length, buffer_size);

    cudaFree(idx2key_ptr);
    cudaFree(decoded_tokens_ptr);
    
    std::vector<std::string> captions = {"hello, world"};


    return captions;
}

TORCH_LIBRARY_IMPL(decode_captions, CUDA, m) {
    m.impl("decodeBatchedCaptions", &decodeBatchedCaptions_cuda);
}

}