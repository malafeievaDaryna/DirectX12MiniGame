#include "ParticleSystem.h"

#include <DDSTextureLoader.h>
#include <DirectXHelpers.h>
#include <d3dcompiler.h>
#include <cassert>
#include <random>
#include "d3dx12.h"
#include "Shaders.h"

using namespace Microsoft::WRL;
using namespace DirectX;

namespace {
const char vs_shader[] =
    "cbuffer PerModelConstants : register (b0)\n"
    "{\n"
    "	matrix MVP;\n"
    "	matrix View;\n"
    "	matrix Proj;\n"
    "	float4 LightPos;\n"
    "	float4 LightDir;\n"
    "}\n"
    "struct VertexShaderOutput\n"
    "{\n"
    "	float4 position : SV_POSITION;\n"
    "	float2 uv : TEXCOORD;\n"
    "	float3 positionWorld : TEXCOORD1;\n"
    "};\n"
    "static const float  _scale = 20;\n"
    "static const float2 _UV[4] = {\n"
    "{1.0, 1.0}, \n"
    "{1.0, 0.0},\n"
    "{0.0, 1.0}, \n"
    "{0.0, 0.0}}; \n"
    "VertexShaderOutput VS_main(\n"
    "   float3 instancePos : INSTANCE_POSITION,\n"
    "   float instanceScale : INSTANCE_SCALE,\n"
    "	uint id: SV_VertexID)\n"
    "{\n"
    "	VertexShaderOutput output;\n"
    "   float currentScale = _scale * instanceScale;\n"
    "   float4 cameraSpace_pos = mul(View, float4(instancePos, 1.0));\n"
    "   float3 cameraDir = float3(0.0, 0.0, 0.0) - cameraSpace_pos.xyz;\n"
    "   float3 upDir = float3(0.0, 1.0, 0.0);\n"
    "   /*producing extruding vectors*/\n"
    "   float3 rightShift = normalize(cross(cameraDir, upDir));\n"
    "   float3 leftShift = -rightShift;\n"
    "   /*rotated quad that way to be perpendicular to camera direction*/\n"
    "   float3 billBoardQuad[4] = {\n"
    "   {0.5*leftShift.x, 0.0, 0.5*leftShift.z}, \n"
    "   {0.5*leftShift.x, 1.0, 0.5*leftShift.z},\n"
    "   {0.5*rightShift.x, 0.0, 0.5*rightShift.z}, \n"
    "   {0.5*rightShift.x, 1.0, 0.5*rightShift.z}}; \n"
    "	output.uv = _UV[id];\n"
    "   float3 extruding = billBoardQuad[id];\n"
    "   float4 pos = cameraSpace_pos + float4(currentScale * extruding, 1.0f);\n"
    "   output.position = mul(Proj, pos);\n"
    "   output.positionWorld = instancePos;\n"
    "	return output;\n"
    "}\n";
const char fs_shader[] =
    "cbuffer PerModelConstants : register (b0)\n"
    "{\n"
    "	matrix MVP;\n"
    "	matrix View;\n"
    "	matrix Proj;\n"
    "	float4 LightPos;\n"
    "	float4 LightDir;\n"
    "}\n"
    "Texture2DArray<float4> inputTexture : register(t0);\n"
    "SamplerState     texureSampler : register(s0);\n"
    "struct PixelShaderOutput\n"
    "{\n"
    "	float4 color : SV_Target;\n"
    "};\n" 
    FLASH_LIGHT_CALCULATION
    "PixelShaderOutput PS_main (float4 position : SV_POSITION,\n"
    "				float2 uv : TEXCOORD, float3 positionWorld : TEXCOORD1) : SV_TARGET\n"
    "{\n"
    "	PixelShaderOutput output = (PixelShaderOutput)0;\n"
    "   float attenuation  = getFlashLightAttenuation(LightPos.xyz, LightDir.xyz, positionWorld, 1000.0f);\n"
    "	output.color = attenuation * inputTexture.Sample (texureSampler, float3(uv[0], uv[1], 0));\n"
    "	return output;\n"
    "}\n";
}  // namespace

ParticleSystem::ParticleSystem(ID3D12Device* device, ID3D12CommandQueue* commandQueue,
                               ID3D12GraphicsCommandList* uploadCommandList, const std::string& texture2dName,
                               const uint32_t instancesAmount, const int32_t spreadingMin, const int32_t spreadingMax)
    : mInstancesAmount(instancesAmount), mSpreadingMin(spreadingMin), mSpreadingMax(spreadingMax) {
    assert(mSpreadingMax > mSpreadingMin && mInstancesAmount > 0u);
    assert(device && commandQueue && uploadCommandList && !texture2dName.empty());
    std::string ddsFilePath = constants::TEXTURE_PATH + texture2dName;
    std::wstring ddsFilePathW(ddsFilePath.begin(), ddsFilePath.end());
    ResourceUploadBatch resourceUpload(device);

    mTexture = utils::CreateTexture(device, uploadCommandList, {texture2dName.c_str()});
    
    CreateConstantBuffer(device);
    CreateRootSignature(device);
    CreatePipelineStateObject(device);
    CreateMeshBuffers(device, uploadCommandList);
}

void ParticleSystem::Update(UINT32 currentFrame, const Camera::ViewProj& viewProj, const XMFLOAT4& lightPos,
                            const XMFLOAT4& lightDir) {
    mConstantBufferData.view = viewProj.view;
    mConstantBufferData.proj = viewProj.proj;
    mConstantBufferData.mvp = DirectX::XMMatrixMultiply(viewProj.view, viewProj.proj);
    mConstantBufferData.lightPos = lightPos;
    mConstantBufferData.lightDir = lightDir;

    void* data;
    mConstantBuffers[currentFrame]->Map(0, nullptr, &data);
    memcpy(data, &mConstantBufferData, sizeof(mConstantBufferData));
    mConstantBuffers[currentFrame]->Unmap(0, nullptr);
}

void ParticleSystem::Draw(UINT32 currentFrame, ID3D12GraphicsCommandList* commandList) {
    assert(commandList);

    commandList->SetPipelineState(mPso.Get());
    commandList->SetGraphicsRootSignature(mRootSignature.Get());
    commandList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP);
    // Set slot 1 of our root signature to the constant buffer view
    commandList->SetGraphicsRootConstantBufferView(1, mConstantBuffers[currentFrame]->GetGPUVirtualAddress());

    ID3D12DescriptorHeap* heaps[] = {mTexture.srvDescriptorHeap.Get()};
    commandList->SetDescriptorHeaps(1, heaps);
    // Set slot 0 of our root signature to point to our descriptor heap with
    // the texture SRV
    commandList->SetGraphicsRootDescriptorTable(0, mTexture.srvDescriptorHeap->GetGPUDescriptorHandleForHeapStart());

    commandList->IASetVertexBuffers(0, 1, &mInstanceBufferView);
    commandList->DrawInstanced(4, mInstancesAmount, 0, 0);
}

void ParticleSystem::CreateConstantBuffer(ID3D12Device* device) {
    assert(device);
    for (int i = 0; i < constants::MAX_FRAMES_IN_FLIGHT; ++i) {
        static const auto uploadHeapProperties = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD);
        static const auto constantBufferDesc = CD3DX12_RESOURCE_DESC::Buffer(sizeof(ConstantBuffer));

        device->CreateCommittedResource(&uploadHeapProperties, D3D12_HEAP_FLAG_NONE, &constantBufferDesc,
                                        D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&mConstantBuffers[i]));

        // we set identity matrix as mvp for models
        mConstantBufferData.mvp = DirectX::XMMatrixIdentity();
        mConstantBufferData.view = DirectX::XMMatrixIdentity();
        mConstantBufferData.proj = DirectX::XMMatrixIdentity();
        void* p;
        mConstantBuffers[i]->Map(0, nullptr, &p);
        memcpy(p, &mConstantBufferData, sizeof(mConstantBufferData));
        mConstantBuffers[i]->Unmap(0, nullptr);
    }
}

void ParticleSystem::CreateMeshBuffers(ID3D12Device* device, ID3D12GraphicsCommandList* uploadCommandList) {
    assert(device && uploadCommandList);

    using namespace std::chrono_literals;
    //-----------------------------------------------------//
    // THE BULLET GENERATOR
    std::random_device rd;
    std::mt19937 gen(rd());  // seed the generator
    std::uniform_real_distribution<float> randomFloats(static_cast<float>(mSpreadingMin), static_cast<float>(mSpreadingMax));
    std::uniform_real_distribution<float> randomFloatsScale(0.5f, 1.0f);
    std::default_random_engine generator;

    std::vector<Instance> instances(mInstancesAmount);

    float x, z;
    for (auto& instance : instances) {
        x = randomFloats(generator);
        z = randomFloats(generator);
        instance.pos = DirectX::XMFLOAT3{x, 0.0f, z};
        instance.scale = randomFloatsScale(generator);
    }

    const int uploadBufferSize = sizeof(Instance) * instances.size();
    const auto uploadHeapProperties = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD);
    const auto uploadBufferDesc = CD3DX12_RESOURCE_DESC::Buffer(uploadBufferSize);

    // Create upload buffer on CPU
    device->CreateCommittedResource(&uploadHeapProperties, D3D12_HEAP_FLAG_NONE, &uploadBufferDesc,
                                    D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&mUploadBuffer));

    // Create instance buffer on the GPU
    // HEAP_TYPE_DEFAULT is on GPU, we also initialize with COPY_DEST state
    // so we don't have to transition into this before copying into them
    const auto defaultHeapProperties = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT);

    device->CreateCommittedResource(&defaultHeapProperties, D3D12_HEAP_FLAG_NONE, &uploadBufferDesc,
                                    D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&mInstanceBuffer));

    // Create buffer views
    mInstanceBufferView.BufferLocation = mInstanceBuffer->GetGPUVirtualAddress();
    mInstanceBufferView.SizeInBytes = uploadBufferSize;
    mInstanceBufferView.StrideInBytes = sizeof(Instance);

    // Copy data on CPU into the upload buffer
    void* p;
    mUploadBuffer->Map(0, nullptr, &p);
    ::memcpy(p, instances.data(), uploadBufferSize);
    mUploadBuffer->Unmap(0, nullptr);

    // Copy data from upload buffer on CPU into the instance buffer on
    // the GPU
    uploadCommandList->CopyBufferRegion(mInstanceBuffer.Get(), 0, mUploadBuffer.Get(), 0, uploadBufferSize);

    // Barriers, batch them together
    const CD3DX12_RESOURCE_BARRIER barriers[1] = {
        CD3DX12_RESOURCE_BARRIER::Transition(mInstanceBuffer.Get(), D3D12_RESOURCE_STATE_COPY_DEST,
                                             D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER)};

    uploadCommandList->ResourceBarrier(1, barriers);
}

void ParticleSystem::CreatePipelineStateObject(ID3D12Device* device) {
    assert(device);
    static const D3D12_INPUT_ELEMENT_DESC layout[] = {
        /* Note:  no need to keep vertex attribure, everything in inctance data
        {"POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
        {"TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT, 0, 12, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
        */
        {"INSTANCE_POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D12_INPUT_CLASSIFICATION_PER_INSTANCE_DATA, 1},
        {"INSTANCE_ACCELERATION_VECTOR", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 12, D3D12_INPUT_CLASSIFICATION_PER_INSTANCE_DATA, 1},
        {"INSTANCE_LIFE_DURATION", 0, DXGI_FORMAT_R32_FLOAT, 0, 24, D3D12_INPUT_CLASSIFICATION_PER_INSTANCE_DATA, 1},
        {"INSTANCE_SPEED_FACTOR", 0, DXGI_FORMAT_R32_FLOAT, 0, 28, D3D12_INPUT_CLASSIFICATION_PER_INSTANCE_DATA, 1},
        {"INSTANCE_ALPHA_FACTOR", 0, DXGI_FORMAT_R32_FLOAT, 0, 32, D3D12_INPUT_CLASSIFICATION_PER_INSTANCE_DATA, 1},
        {"INSTANCE_SCALE", 0, DXGI_FORMAT_R32_FLOAT, 0, 36, D3D12_INPUT_CLASSIFICATION_PER_INSTANCE_DATA, 1},
    };

#if defined(_DEBUG)
    // Enable better shader debugging with the graphics debugging tools.
    UINT compileFlags = D3DCOMPILE_DEBUG | D3DCOMPILE_SKIP_OPTIMIZATION;
#else
    UINT compileFlags = 0;
#endif

    ID3D10Blob* errorMsgs{nullptr};

    ComPtr<ID3DBlob> vertexShader;
    auto res = D3DCompile(vs_shader, sizeof(vs_shader), "", nullptr, nullptr, "VS_main", "vs_5_1", compileFlags, 0, &vertexShader,
               &errorMsgs);
    utils::ThrowIfFailed(res, errorMsgs ? (const char*)errorMsgs->GetBufferPointer() : nullptr);

    ComPtr<ID3DBlob> pixelShader;
    res = D3DCompile(fs_shader, sizeof(fs_shader), "", nullptr, nullptr, "PS_main", "ps_5_1", compileFlags, 0, &pixelShader,
               &errorMsgs);
    utils::ThrowIfFailed(res, errorMsgs ? (const char*)errorMsgs->GetBufferPointer() : nullptr);

    D3D12_GRAPHICS_PIPELINE_STATE_DESC psoDesc = {};
    psoDesc.VS.BytecodeLength = vertexShader->GetBufferSize();
    psoDesc.VS.pShaderBytecode = vertexShader->GetBufferPointer();
    psoDesc.PS.BytecodeLength = pixelShader->GetBufferSize();
    psoDesc.PS.pShaderBytecode = pixelShader->GetBufferPointer();
    psoDesc.pRootSignature = mRootSignature.Get();
    psoDesc.NumRenderTargets = 1;
    psoDesc.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM_SRGB;
    psoDesc.DSVFormat = constants::DEPTH_FORMAT;
    psoDesc.InputLayout.NumElements = std::extent<decltype(layout)>::value;
    psoDesc.InputLayout.pInputElementDescs = layout;
    psoDesc.RasterizerState = CD3DX12_RASTERIZER_DESC(D3D12_DEFAULT);
    psoDesc.RasterizerState.CullMode = D3D12_CULL_MODE_BACK;
    psoDesc.RasterizerState.FrontCounterClockwise = FALSE;
    //psoDesc.RasterizerState.CullMode = D3D12_CULL_MODE_NONE; // both faces drawn
    psoDesc.BlendState = CD3DX12_BLEND_DESC(D3D12_DEFAULT);
    // Simple alpha blending
    psoDesc.BlendState.RenderTarget[0].BlendEnable = true;
    psoDesc.BlendState.RenderTarget[0].SrcBlend = D3D12_BLEND_SRC_ALPHA;
    psoDesc.BlendState.RenderTarget[0].DestBlend = D3D12_BLEND_INV_SRC_ALPHA;
    psoDesc.BlendState.RenderTarget[0].BlendOp = D3D12_BLEND_OP_ADD;
    psoDesc.BlendState.RenderTarget[0].SrcBlendAlpha = D3D12_BLEND_ONE;
    psoDesc.BlendState.RenderTarget[0].DestBlendAlpha = D3D12_BLEND_ZERO;
    psoDesc.BlendState.RenderTarget[0].BlendOpAlpha = D3D12_BLEND_OP_ADD;
    psoDesc.BlendState.RenderTarget[0].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
    psoDesc.SampleDesc.Count = 1;
    psoDesc.SampleMask = 0xFFFFFFFF;
    psoDesc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    
    psoDesc.DepthStencilState = CD3DX12_DEPTH_STENCIL_DESC(D3D12_DEFAULT);
    psoDesc.DepthStencilState.StencilEnable = false;
    psoDesc.DepthStencilState.DepthEnable = false;
    psoDesc.DepthStencilState.DepthFunc = D3D12_COMPARISON_FUNC_LESS;  // we accept depth value 1.0  of our edges as well
    psoDesc.DepthStencilState.DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ALL;  // no need to write into depth

    device->CreateGraphicsPipelineState(&psoDesc, IID_PPV_ARGS(&mPso));
}

void ParticleSystem::CreateRootSignature(ID3D12Device* device) {
    assert(device);
    // We have two root parameters, one is a pointer to a descriptor heap
    // with a SRV, the second is a constant buffer view
    CD3DX12_ROOT_PARAMETER parameters[2];

    // Create a descriptor table with one entry in our descriptor heap
    CD3DX12_DESCRIPTOR_RANGE range{D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0};
    parameters[0].InitAsDescriptorTable(1, &range);

    // Our constant buffer view
    parameters[1].InitAsConstantBufferView(0, 0, D3D12_SHADER_VISIBILITY_ALL);

    // We don't use another descriptor heap for the sampler, instead we use a
    // static sampler
    CD3DX12_STATIC_SAMPLER_DESC samplers[1];
    samplers[0].Init(0, D3D12_FILTER_MIN_MAG_LINEAR_MIP_POINT);

    CD3DX12_ROOT_SIGNATURE_DESC descRootSignature;

    // Create the root signature
    descRootSignature.Init(2, parameters, 1, samplers, D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT);

    ComPtr<ID3DBlob> rootBlob;
    ComPtr<ID3DBlob> errorBlob;
    D3D12SerializeRootSignature(&descRootSignature, D3D_ROOT_SIGNATURE_VERSION_1, &rootBlob, &errorBlob);

    device->CreateRootSignature(0, rootBlob->GetBufferPointer(), rootBlob->GetBufferSize(), IID_PPV_ARGS(&mRootSignature));
}
