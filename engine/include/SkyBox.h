#pragma once

#include <DescriptorHeap.h>
#include <ResourceUploadBatch.h>
#include <d3d12.h>
#include <directxmath.h>
#include <memory.h>
#include <wrl.h>
#include <string>

#include "Camera.h"
#include "Utils.h"

class SkyBox {
    struct ConstantBuffer {
        DirectX::XMMATRIX mvp;
    };

public:
    SkyBox(ID3D12Device* device, ID3D12CommandQueue* commandQueue, ID3D12GraphicsCommandList* uploadCommandList,
           const std::string& ddsFileName);
    void Update(UINT32 currentFrame, const Camera::ViewProj& viewProj);
    void Draw(UINT32 currentFrame, ID3D12GraphicsCommandList* commandList);

private:
    void CreateRootSignature(ID3D12Device* device);
    void CreatePipelineStateObject(ID3D12Device* device);
    void CreateMeshBuffers(ID3D12Device* device, ID3D12GraphicsCommandList* uploadCommandList);
    void CreateConstantBuffer(ID3D12Device* device);

private:
    // TEXTURE HANDLERS
    Microsoft::WRL::ComPtr<ID3D12Resource> mCubemap;
    std::unique_ptr<DirectX::DescriptorHeap> mSrvDescriptorHeap;

    // MESH BUFFERS
    Microsoft::WRL::ComPtr<ID3D12Resource> mUploadBuffer{};
    Microsoft::WRL::ComPtr<ID3D12Resource> mVertexBuffer{};
    D3D12_VERTEX_BUFFER_VIEW mVertexBufferView;
    Microsoft::WRL::ComPtr<ID3D12Resource> mIndexBuffer{};
    D3D12_INDEX_BUFFER_VIEW mIndexBufferView{};

    // MVP matrix
    ConstantBuffer mConstantBufferData{};
    Microsoft::WRL::ComPtr<ID3D12Resource> mConstantBuffers[constants::MAX_FRAMES_IN_FLIGHT];

    // PIPELINE&SHADER
    Microsoft::WRL::ComPtr<ID3D12RootSignature> mRootSignature{};
    Microsoft::WRL::ComPtr<ID3D12PipelineState> mPso{};
};
