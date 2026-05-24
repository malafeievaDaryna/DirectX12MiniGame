#include "SkyBox.h"

#include <DDSTextureLoader.h>
#include <DirectXHelpers.h>
#include <d3dcompiler.h>
#include <cassert>
#include "Utils.h"
#include "d3dx12.h"

using namespace Microsoft::WRL;
using namespace DirectX;

namespace {
const char vs_shader[] =
    "cbuffer PerModelConstants : register (b0)\n"
    "{\n"
    "	matrix MVP;\n"
    "   float T;\n"
    "}\n"
    "struct VertexShaderOutput\n"
    "{\n"
    "	float4 position : SV_POSITION;\n"
    "	float3 worldDir : TEXCOORD;\n"
    "};\n"
    "VertexShaderOutput VS_main(float3 position : POSITION)\n"
    "{\n"
    "	VertexShaderOutput output;\n"
    "   output.position = mul(MVP, float4(position, 1));\n"
    "   output.position.z = output.position.w; // Ensure depth is at far plane\n"
    "   // Adjusting Y for visual framing\n"
    "   output.position.y = 0.3 * output.position.y + 0.15; // for better visual effect [-1: 1] -> [-0.15; 0.45] \n"
    "	output.worldDir = position.xyz;\n"
    "	return output;\n"
    "}\n";

const char fs_shader_simple_skyBox[] =
    "TextureCube<float4> inputTexture : register(t0);\n"
    "SamplerState     texureSampler : register(s0);\n"
    "float4 PS_main (float4 position : SV_POSITION,\n"
    "				float3 uv : TEXCOORD) : SV_TARGET\n"
    "{\n"
    "   float attenuation = 0.03;"
    "	return float4(attenuation, attenuation, attenuation, 1.0) * inputTexture.Sample (texureSampler, "
    "normalize(uv));"
    "}\n";
const char fs_shader[] =
    "cbuffer PerModelConstants : register (b0)\n"
    "{\n"
    "    matrix MVP;\n"
    "    float T;\n"
    "}\n"
    "\n"
    "// Simple hash function for randomness\n"
    "float hash(float2 p) {\n"
    "    return frac(sin(dot(p, float2(12.9898, 78.233))) * 43758.5453123);\n"
    "}\n"
    "\n"
    "// 2D Value Noise\n"
    "float noise(float2 p) {\n"
    "    float2 i = floor(p);\n"
    "    float2 f = frac(p);\n"
    "    f = f * f * (3.0 - 2.0 * f);\n"
    "    return lerp(lerp(hash(i + float2(0,0)), hash(i + float2(1,0)), f.x),\n"
    "                lerp(hash(i + float2(0,1)), hash(i + float2(1,1)), f.x), f.y);\n"
    "}\n"
    "\n"
    "// Fractional Brownian Motion with ridge-like patterns for lava texture\n"
    "float fbm_lava(float2 p) {\n"
    "    float v = 0.0;\n"
    "    float amp = 0.5;\n"
    "    for (int i = 0; i < 6; i++) {\n"
    "        // Use absolute noise for filament/vein structures\n"
    "        v += (1.0 - abs(noise(p) * 2.0 - 1.0)) * amp;\n"
    "        p *= 2.2;\n"
    "        amp *= 0.5;\n"
    "    }\n"
    "    return v;\n"
    "}\n"
    "\n"
    "float4 PS_main(float4 position : SV_POSITION, float3 worldDir : TEXCOORD) : SV_TARGET\n"
    "{\n"
    "    float3 dir = normalize(worldDir);\n"
    "    \n"
    "    // Project 3D direction to 2D space for the sky dome\n"
    "    float2 uv = dir.xz / (abs(dir.y) + 0.4);\n"
    "    \n"
    "    // Lava breathing effect using sine wave\n"
    "    float pulse = sin(T * 1.5) * 0.5 + 0.5;\n"
    "    \n"
    "    // Add turbulence through domain warping\n"
    "    float t = T * 0.2;\n"
    "    float2 q = float2(fbm_lava(uv + t), fbm_lava(uv + float2(1.2, 0.5)));\n"
    "    float n = fbm_lava(uv + q + t);\n"
    "\n"
    "    // --- HELLISH LAVA PALETTE ---\n"
    "    float3 smokeBlack = float3(0.05, 0.0, 0.0);   // Dark volcanic smoke\n"
    "    float3 lavaRed    = float3(0.9, 0.0, 0.0);    // Blood-red lava core\n"
    "    float3 lavaOrange = float3(1.0, 0.35, 0.0);    // Molten orange flow\n"
    "    float3 lavaHot    = float3(1.0, 0.8, 0.3);    // Incandescent yellow hotspots\n"
    "\n"
    "    float3 col = lerp(smokeBlack, lavaRed, n);\n"
    "    // Intense glow points that react to the 'pulse'\n"
    "    float fireCore = pow(abs(n), 3.0 + pulse * 1.0);\n"
    "    col = lerp(col, lavaOrange, fireCore);\n"
    "    col = lerp(col, lavaHot, pow(abs(n), 7.0 + pulse * 2.0));\n"
    "\n"
    "    // Master brightness with breathing intensity\n"
    "    float intensity = 1.2 + pulse * 0.6;\n"
    "    col *= n * intensity;\n"
    "\n"
    "    // Atmospheric horizon glow\n"
    "    col += lavaRed * 0.2 * (1.0 - abs(dir.y)) * (0.8 + pulse * 0.2);\n"
    "\n"
    "    // Gamma correction and final color clamping\n"
    "    col = pow(abs(col), 0.5);\n"
    "\n"
    "    return float4(col, 1.0);\n"
    "}\n";
}  // namespace

SkyBox::SkyBox(ID3D12Device* device, ID3D12CommandQueue* commandQueue, ID3D12GraphicsCommandList* uploadCommandList,
               const std::string& ddsFileName) {
    assert(device && commandQueue && uploadCommandList && !ddsFileName.empty());
    std::string ddsFilePath = constants::TEXTURE_PATH + ddsFileName;
    std::wstring ddsFilePathW(ddsFilePath.begin(), ddsFilePath.end());
    ResourceUploadBatch resourceUpload(device);

    mSrvDescriptorHeap = std::make_unique<DescriptorHeap>(device, 1);

    resourceUpload.Begin();

    utils::ThrowIfFailed(
        DirectX::CreateDDSTextureFromFile(device, resourceUpload, ddsFilePathW.c_str(), mCubemap.ReleaseAndGetAddressOf()));

    CreateShaderResourceView(device, mCubemap.Get(), mSrvDescriptorHeap->GetFirstCpuHandle(), true);

    auto uploadResourcesFinished = resourceUpload.End(commandQueue);

    uploadResourcesFinished.wait();

    mAccumulatedTimeS = 0.0f;
    CreateConstantBuffer(device);
    CreateRootSignature(device);
    CreatePipelineStateObject(device);
    CreateMeshBuffers(device, uploadCommandList);
}

void SkyBox::Update(UINT32 currentFrame, const Camera::ViewProj& viewProj, float frameTimeMS) {
    /**
    * NOTE: rotating skybox
    static float angle = 0.0f;
    angle += 0.01f;
    const static DirectX::XMVECTOR rotationAxis = DirectX::XMVectorSet(0, 1, 0, 0);
    auto rotation = DirectX::XMMatrixRotationAxis(rotationAxis, DirectX::XMConvertToRadians(angle));
    */

    // multiplication by projection matrix will mask distinct edges of the cube
    XMMATRIX viewRotationOnly = utils::extractRotationMatrix(viewProj.view);
    mConstantBufferData.mvp = DirectX::XMMatrixMultiply(viewRotationOnly, viewProj.proj);

    // our animation is designed for 144 fps, and we use seconds
    float frameTimeFactorSec = (frameTimeMS / constants::_144_FPS_TO_MS) * 0.001f;
    mAccumulatedTimeS += frameTimeFactorSec;
    mConstantBufferData.T = mAccumulatedTimeS;

    void* data;
    mConstantBuffers[currentFrame]->Map(0, nullptr, &data);
    memcpy(data, &mConstantBufferData, sizeof(mConstantBufferData));
    mConstantBuffers[currentFrame]->Unmap(0, nullptr);
}

void SkyBox::Draw(UINT32 currentFrame, ID3D12GraphicsCommandList* commandList) {
    assert(commandList);

    commandList->SetPipelineState(mPso.Get());
    commandList->SetGraphicsRootSignature(mRootSignature.Get());
    commandList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    // Set slot 1 of our root signature to the constant buffer view
    commandList->SetGraphicsRootConstantBufferView(1, mConstantBuffers[currentFrame]->GetGPUVirtualAddress());

    ID3D12DescriptorHeap* heaps[] = {mSrvDescriptorHeap->Heap()};
    commandList->SetDescriptorHeaps(1, heaps);
    // Set slot 0 of our root signature to point to our descriptor heap with
    // the texture SRV
    commandList->SetGraphicsRootDescriptorTable(0, mSrvDescriptorHeap->GetFirstGpuHandle());

    commandList->IASetVertexBuffers(0, 1, &mVertexBufferView);
    commandList->IASetIndexBuffer(&mIndexBufferView);
    commandList->DrawIndexedInstanced(36, 1, 0, 0, 0);
}

void SkyBox::CreateConstantBuffer(ID3D12Device* device) {
    assert(device);
    for (int i = 0; i < constants::MAX_FRAMES_IN_FLIGHT; ++i) {
        static const auto uploadHeapProperties = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD);
        static const auto constantBufferDesc = CD3DX12_RESOURCE_DESC::Buffer(sizeof(ConstantBuffer));

        device->CreateCommittedResource(&uploadHeapProperties, D3D12_HEAP_FLAG_NONE, &constantBufferDesc,
                                        D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&mConstantBuffers[i]));

        // we set identity matrix as mvp for models and zero time
        mConstantBufferData.mvp = DirectX::XMMatrixIdentity();
        mConstantBufferData.T = 0.0f;
        void* p;
        mConstantBuffers[i]->Map(0, nullptr, &p);
        memcpy(p, &mConstantBufferData, sizeof(mConstantBufferData));
        mConstantBuffers[i]->Unmap(0, nullptr);
    }
}

void SkyBox::CreateMeshBuffers(ID3D12Device* device, ID3D12GraphicsCommandList* uploadCommandList) {
    assert(device && uploadCommandList);
    struct Vertex {
        float position[3];
    };

    /// Note: designed for PRIMITIVE_TOPOLOGY_TRIANGLE_LIST and FrontCounterClockwise order
    const Vertex vertices[] = {// front
                               {{-1.0, -1.0, 1.0}},
                               {{1.0, -1.0, 1.0}},
                               {{1.0, 1.0, 1.0}},
                               {{-1.0, 1.0, 1.0}},
                               // back
                               {{-1.0, -1.0, -1.0}},
                               {{1.0, -1.0, -1.0}},
                               {{1.0, 1.0, -1.0}},
                               {{-1.0, 1.0, -1.0}}};

    const uint32_t indices[] = {// front
                                0, 1, 2, 2, 3, 0,
                                // right
                                1, 5, 6, 6, 2, 1,
                                // back
                                6, 5, 4, 4, 7, 6,
                                // left
                                4, 0, 3, 3, 7, 4,
                                // bottom
                                5, 4, 0, 0, 1, 5,  // rejected by front face mode if no need to be drawn
                                                   // 0, 4, 5, 5, 1, 0, // accepted by front face mode
                                                   // top
                                6, 3, 2, 6, 7, 3};

    const int uploadBufferSize = sizeof(vertices) + sizeof(indices);
    const auto uploadHeapProperties = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD);
    const auto uploadBufferDesc = CD3DX12_RESOURCE_DESC::Buffer(uploadBufferSize);

    // Create upload buffer on CPU
    device->CreateCommittedResource(&uploadHeapProperties, D3D12_HEAP_FLAG_NONE, &uploadBufferDesc,
                                    D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&mUploadBuffer));

    // Create vertex & index buffer on the GPU
    // HEAP_TYPE_DEFAULT is on GPU, we also initialize with COPY_DEST state
    // so we don't have to transition into this before copying into them
    const auto defaultHeapProperties = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT);

    const auto vertexBufferDesc = CD3DX12_RESOURCE_DESC::Buffer(sizeof(vertices));
    device->CreateCommittedResource(&defaultHeapProperties, D3D12_HEAP_FLAG_NONE, &vertexBufferDesc,
                                    D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&mVertexBuffer));

    const auto indexBufferDesc = CD3DX12_RESOURCE_DESC::Buffer(sizeof(indices));
    device->CreateCommittedResource(&defaultHeapProperties, D3D12_HEAP_FLAG_NONE, &indexBufferDesc,
                                    D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&mIndexBuffer));

    // Create buffer views
    mVertexBufferView.BufferLocation = mVertexBuffer->GetGPUVirtualAddress();
    mVertexBufferView.SizeInBytes = sizeof(vertices);
    mVertexBufferView.StrideInBytes = sizeof(Vertex);

    mIndexBufferView.BufferLocation = mIndexBuffer->GetGPUVirtualAddress();
    mIndexBufferView.SizeInBytes = sizeof(indices);
    mIndexBufferView.Format = DXGI_FORMAT_R32_UINT;

    // Copy data on CPU into the upload buffer
    void* p;
    mUploadBuffer->Map(0, nullptr, &p);
    ::memcpy(p, vertices, sizeof(vertices));
    ::memcpy(static_cast<unsigned char*>(p) + sizeof(vertices), indices, sizeof(indices));
    mUploadBuffer->Unmap(0, nullptr);

    // Copy data from upload buffer on CPU into the index/vertex buffer on
    // the GPU
    uploadCommandList->CopyBufferRegion(mVertexBuffer.Get(), 0, mUploadBuffer.Get(), 0, sizeof(vertices));
    uploadCommandList->CopyBufferRegion(mIndexBuffer.Get(), 0, mUploadBuffer.Get(), sizeof(vertices), sizeof(indices));

    // Barriers, batch them together
    const CD3DX12_RESOURCE_BARRIER barriers[2] = {
        CD3DX12_RESOURCE_BARRIER::Transition(mVertexBuffer.Get(), D3D12_RESOURCE_STATE_COPY_DEST,
                                             D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER),
        CD3DX12_RESOURCE_BARRIER::Transition(mIndexBuffer.Get(), D3D12_RESOURCE_STATE_COPY_DEST,
                                             D3D12_RESOURCE_STATE_INDEX_BUFFER)};

    uploadCommandList->ResourceBarrier(2, barriers);
}

void SkyBox::CreatePipelineStateObject(ID3D12Device* device) {
    assert(device);
    static const D3D12_INPUT_ELEMENT_DESC layout[] = {
        {"POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0}};

#if defined(_DEBUG)
    // Enable better shader debugging with the graphics debugging tools.
    UINT compileFlags = D3DCOMPILE_DEBUG | D3DCOMPILE_SKIP_OPTIMIZATION;
#else
    UINT compileFlags = 0;
#endif

    ComPtr<ID3DBlob> vertexShader;
    D3DCompile(vs_shader, sizeof(vs_shader), "", nullptr, nullptr, "VS_main", "vs_5_1", compileFlags, 0, &vertexShader, nullptr);

    ComPtr<ID3DBlob> pixelShader;
    D3DCompile(fs_shader, sizeof(fs_shader), "", nullptr, nullptr, "PS_main", "ps_5_1", compileFlags, 0, &pixelShader, nullptr);

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
    psoDesc.RasterizerState.FrontCounterClockwise = TRUE;  // Front face order is CounterClockwise!
    psoDesc.BlendState = CD3DX12_BLEND_DESC(D3D12_DEFAULT);
    // disable blending
    psoDesc.BlendState.RenderTarget[0].BlendEnable = FALSE;
    psoDesc.SampleDesc.Count = 1;
    psoDesc.SampleMask = 0xFFFFFFFF;
    psoDesc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;

    psoDesc.DepthStencilState = CD3DX12_DEPTH_STENCIL_DESC(D3D12_DEFAULT);
    psoDesc.DepthStencilState.StencilEnable = false;
    psoDesc.DepthStencilState.DepthEnable = TRUE;
    psoDesc.DepthStencilState.DepthFunc = D3D12_COMPARISON_FUNC_LESS_EQUAL;  // we accept depth value 1.0  of our edges as well
    psoDesc.DepthStencilState.DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ZERO;  // no need to write into depth

    device->CreateGraphicsPipelineState(&psoDesc, IID_PPV_ARGS(&mPso));
}

void SkyBox::CreateRootSignature(ID3D12Device* device) {
    assert(device);
    // We have two root parameters, one is a pointer to a descriptor heap
    // with a SRV, the second is a constant buffer view
    CD3DX12_ROOT_PARAMETER parameters[2];

    // Create a descriptor table with one entry in our descriptor heap
    CD3DX12_DESCRIPTOR_RANGE range{D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0};
    parameters[0].InitAsDescriptorTable(1, &range);

    // Our constant buffer view (it's visible to all shaders, as T is read in PS)
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
