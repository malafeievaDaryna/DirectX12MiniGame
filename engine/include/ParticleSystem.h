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

class ParticleSystem {
    struct ConstantBuffer {
        DirectX::XMMATRIX mvp;
        DirectX::XMMATRIX view;
        DirectX::XMMATRIX proj;
        DirectX::XMFLOAT4 lightPos;
        DirectX::XMFLOAT4 lightDir;
    };

    struct Instance {
        DirectX::XMFLOAT3 pos{0.0f, 0.0f, 0.0f};
        DirectX::XMFLOAT3 acceleration{0.0f, 0.0f, 0.0f};
        float lifeDuration{1.0f};  // ms allocated for life of particle
        float speedK{1.0f};
        float alphaK{1.0f};
        float scale{1.0f};
    };

public:
    ParticleSystem(ID3D12Device* device, ID3D12CommandQueue* commandQueue, ID3D12GraphicsCommandList* uploadCommandList,
                   const std::string& texture2dName, const uint32_t instancesAmount, const int32_t spreadingMin = 0,
                   const int32_t spreadingMax = 1);
    void Update(UINT32 currentFrame, const Camera::ViewProj& viewProj, const DirectX::XMFLOAT4& lightPos,
                const DirectX::XMFLOAT4& lightDir);
    void Draw(UINT32 currentFrame, ID3D12GraphicsCommandList* commandList);

private:
    void CreateRootSignature(ID3D12Device* device);
    void CreatePipelineStateObject(ID3D12Device* device);
    void CreateMeshBuffers(ID3D12Device* device, ID3D12GraphicsCommandList* uploadCommandList);
    void CreateConstantBuffer(ID3D12Device* device);

private:
    // TEXTURE HANDLERS
    utils::Texture2DResource mTexture;

    // MESH BUFFERS
    Microsoft::WRL::ComPtr<ID3D12Resource> mUploadBuffer{};
    Microsoft::WRL::ComPtr<ID3D12Resource> mInstanceBuffer{};
    D3D12_VERTEX_BUFFER_VIEW mInstanceBufferView;

    // MVP matrix
    ConstantBuffer mConstantBufferData{};
    Microsoft::WRL::ComPtr<ID3D12Resource> mConstantBuffers[constants::MAX_FRAMES_IN_FLIGHT];

    // PIPELINE&SHADER
    Microsoft::WRL::ComPtr<ID3D12RootSignature> mRootSignature{};
    Microsoft::WRL::ComPtr<ID3D12PipelineState> mPso{};

    const uint32_t mInstancesAmount{0u};
    const int32_t mSpreadingMin{0};
    const int32_t mSpreadingMax{1};
};
