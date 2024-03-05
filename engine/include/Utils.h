#pragma once

#include <wrl.h>
#include <iostream>
#include <string>

class ID3D12GraphicsCommandList;
class ID3D12Device;
class ID3D12DescriptorHeap;
class ID3D12Resource;
namespace utils {
struct Texture2DResource {
    Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> srvDescriptorHeap;
    Microsoft::WRL::ComPtr<ID3D12Resource> image;
    Microsoft::WRL::ComPtr<ID3D12Resource> stagingBuffer;
};
static const std::string TEXTURE_PATH = "textures\\";
template <typename... Args>
void log_info(Args... args) {
    ((std::cout << " " << args), ...) << std::endl;
}
template <typename... Args>
void log_err(Args... args) {
    ((std::cerr << " " << args), ...) << std::endl;
    std::exit(-1);
}
#ifdef NDEBUG
#define log_debug(...) ((void)0)
#else
template <typename... Args>
void log_debug(Args... args) {
    log_info(args...);
}
#endif

Texture2DResource CreateTexture(ID3D12Device* device, ID3D12GraphicsCommandList* uploadCommandList,
                                const std::string& textureFileName);
}  // namespace utils
