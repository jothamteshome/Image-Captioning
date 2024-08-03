#include <string>
#include <vector>

#include <iostream>

#include <ATen/ATen.h>
#include <torch/extension.h>
#include <torch/library.h>


namespace decode_captions {

std::string join(const std::vector<std::string>& separated_string) {
    std::string joined_string = separated_string[0];

    for (int i = 1; i < separated_string.size(); i++) {
        joined_string = joined_string + " " + separated_string[i];
    }

    return joined_string;
}


std::string decodeCaption(const at::TensorAccessor<int64_t, 1>& accessor, const std::vector<std::string>& idx2key) {
    std::vector<std::string> tokenized_caption;

    for (int i = 0; i < accessor.size(0); i++) {
        int token_id = accessor[i];
        std::string token = idx2key[token_id];

        if (token != "<start>" && token != "<end>") {
            tokenized_caption.push_back(token);
        }

        if (token == "<end>") {
            break;
        }
    }

    return join(tokenized_caption);
}


std::vector<std::string> decodeBatchedCaptions_cpu(const at::Tensor& encoded_captions, const std::vector<std::string>& idx2key, const int64_t& max_token_length) {
    TORCH_CHECK(encoded_captions.dtype() == at::kLong, "Input tensor must have dtype Long");
    TORCH_CHECK(encoded_captions.device().type() == at::DeviceType::CPU, "Input tensor must be on CPU");

    std::cout << "CPU" << std::endl;

    std::vector<std::string> captions;

    auto caption_accessor = encoded_captions.accessor<int64_t,2>();

    for (int i = 0; i < caption_accessor.size(0); i++) {
        captions.push_back(decodeCaption(caption_accessor[i], idx2key));
    }

    return captions;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}

TORCH_LIBRARY(decode_captions, m) {
    m.def("decodeBatchedCaptions(Tensor encoded_captions, vector<string> idx2key, int max_token_length) -> vector<string>");
}

TORCH_LIBRARY_IMPL(decode_captions, CPU, m) {
    m.impl("decodeBatchedCaptions", &decodeBatchedCaptions_cpu);
}

}