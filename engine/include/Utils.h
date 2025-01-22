#pragma once

#include <directxmath.h>
#include <dxgi.h>
#include <wrl.h>
#include <iostream>
#include <string>

class ID3D12GraphicsCommandList;
class ID3D12Device;
class ID3D12DescriptorHeap;
class ID3D12Resource;

namespace constants {
static constexpr uint32_t MAX_FRAMES_IN_FLIGHT = 3;
static constexpr DXGI_FORMAT DEPTH_FORMAT = DXGI_FORMAT_D32_FLOAT;
static const std::string TEXTURE_PATH = "textures\\";
}  // namespace constants

namespace utils {
struct Texture2DResource {
    Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> srvDescriptorHeap;
    Microsoft::WRL::ComPtr<ID3D12Resource> image;
    Microsoft::WRL::ComPtr<ID3D12Resource> stagingBuffer;
};
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

void ThrowIfFailed(HRESULT hr);

DirectX::XMMATRIX extractRotationMatrix(const DirectX::XMMATRIX& input);

Texture2DResource CreateTexture(ID3D12Device* device, ID3D12GraphicsCommandList* uploadCommandList,
                                const std::string& textureFileName);
}  // namespace utils
