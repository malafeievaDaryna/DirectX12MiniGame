#pragma once

#define NOMINMAX

#include <d3d12.h>
#include <dxgi.h>
#include <wrl.h>  // ComPtr template (kinda smart pointers for COM objects)
#include <memory>
#include <string>

class Window;

class DirectXRenderer {
    static constexpr uint32_t MAX_FRAMES_IN_FLIGHT = 3;  /// triple buffering to maximize performance
public:
    DirectXRenderer();
    ~DirectXRenderer();

    void Initialize(const std::string& title, int width, int height);
    bool Run();

private:
    void Shutdown();
    void Render();
    bool CheckTearingSupport();
    void CreateMeshBuffers(ID3D12GraphicsCommandList* uploadCommandList);
    void CreateRootSignature();
    void CreatePipelineStateObject();
    void CreateDeviceAndSwapChain();
    void SetupSwapChain();
    void SetupRenderTargets();

private:
    std::unique_ptr<Window, void (*)(Window*)> mWindow;
    D3D12_VIEWPORT mViewport{};
    D3D12_RECT mRectScissor{};
    Microsoft::WRL::ComPtr<IDXGISwapChain> mSwapChain{};
    Microsoft::WRL::ComPtr<ID3D12Device> mDevice{};
    Microsoft::WRL::ComPtr<ID3D12Resource> mRenderTargets[MAX_FRAMES_IN_FLIGHT]{};
    Microsoft::WRL::ComPtr<ID3D12CommandQueue> mCommandQueue{};

    HANDLE mFrameFenceEvents[MAX_FRAMES_IN_FLIGHT]{nullptr};
    Microsoft::WRL::ComPtr<ID3D12Fence> mFrameFences[MAX_FRAMES_IN_FLIGHT]{};
    UINT64 mCurrentFenceValue{0u};
    UINT64 mFenceValues[MAX_FRAMES_IN_FLIGHT]{};
    UINT32 m_currentFrame{0u};

    Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> mRenderTargetDescriptorHeap{};
    UINT64 mRenderTargetViewDescriptorSize{0u};

    Microsoft::WRL::ComPtr<ID3D12RootSignature> mRootSignature{};
    Microsoft::WRL::ComPtr<ID3D12PipelineState> mPso{};

    Microsoft::WRL::ComPtr<ID3D12Resource> mUploadBuffer{};
    Microsoft::WRL::ComPtr<ID3D12Resource> mVertexBuffer{};
    D3D12_VERTEX_BUFFER_VIEW mVertexBufferView;

    Microsoft::WRL::ComPtr<ID3D12Resource> mIndexBuffer{};
    D3D12_INDEX_BUFFER_VIEW mIndexBufferView{};
    Microsoft::WRL::ComPtr<ID3D12CommandAllocator> mCommandAllocators[MAX_FRAMES_IN_FLIGHT]{};
    Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList> mCommandLists[MAX_FRAMES_IN_FLIGHT]{};
};
