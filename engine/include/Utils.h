#pragma once

#include <d3d12.h>  // D3D12_GPU_DESCRIPTOR_HANDLE for Free fun requires this
#include <directxmath.h>
#include <dxgi.h>
#include <wrl.h>
#include <iostream>
#include <string>
#include <vector>

class ID3D12GraphicsCommandList;
class ID3D12Device;
class ID3D12DescriptorHeap;
class ID3D12Resource;

namespace constants {
static constexpr uint32_t MAX_FRAMES_IN_FLIGHT = 3;
static constexpr DXGI_FORMAT DEPTH_FORMAT = DXGI_FORMAT_D32_FLOAT;
static constexpr DXGI_FORMAT RENDER_TARGET_FORMAT = DXGI_FORMAT_R8G8B8A8_UNORM_SRGB;
static const std::string TEXTURE_PATH = "textures\\";
const static float _144_FPS_TO_MS = 6.94f;
}  // namespace constants

namespace utils {
// Simple free list based allocator
class DescriptorHeapAllocator {
public:
    DescriptorHeapAllocator(ID3D12Device* device, ID3D12DescriptorHeap* bigBaseHeap);
    void Destroy();
    void Alloc(D3D12_CPU_DESCRIPTOR_HANDLE* out_cpu_desc_handle,
               D3D12_GPU_DESCRIPTOR_HANDLE* out_gpu_desc_handle);  // used as callbac
    void Free(D3D12_CPU_DESCRIPTOR_HANDLE out_cpu_desc_handle,
              D3D12_GPU_DESCRIPTOR_HANDLE out_gpu_desc_handle);  // used as callbac

    DescriptorHeapAllocator(const DescriptorHeapAllocator&) = delete;
    DescriptorHeapAllocator(DescriptorHeapAllocator&&) = delete;

private:
    ID3D12DescriptorHeap* mHeap = nullptr;
    D3D12_DESCRIPTOR_HEAP_TYPE mHeapType = D3D12_DESCRIPTOR_HEAP_TYPE_NUM_TYPES;
    D3D12_CPU_DESCRIPTOR_HANDLE mHeapStartCpu;
    D3D12_GPU_DESCRIPTOR_HANDLE mHeapStartGpu;
    UINT mHeapHandleIncrement;
    std::vector<int32_t> mFreeIndices;
};

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

void ThrowIfFailed(HRESULT hr, const char* msg = nullptr);

DirectX::XMMATRIX extractRotationMatrix(const DirectX::XMMATRIX& input);

Texture2DResource CreateTexture(ID3D12Device* device, ID3D12GraphicsCommandList* uploadCommandList,
                                const std::string& textureFileName);
}  // namespace utils
