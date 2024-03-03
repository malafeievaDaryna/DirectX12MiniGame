#include "DirectXRenderer.h"
#include "Window.h"
/**
 * d3dx12.h provides some useful classes that will simplify some of the functions
 * it needs to be downloaded separately from the Microsoft DirectX repository
 * (https://github.com/Microsoft/DirectX-Graphics-Samples/tree/master/Libraries/D3DX12)
 */
#include "Shaders.h"
#include "d3dx12.h"

#include <d3dcompiler.h>
#include <dxgi1_5.h>
#include <algorithm>
#include <cassert>
#include <iostream>
#include <WICTextureLoader.h>
#include <ResourceUploadBatch.h >
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

template <typename... Args>
void log_info(Args... args) {
    ((std::cout << " " << args), ...) << std::endl;
}
template <typename... Args>
void log_err(Args... args) {
    ((std::cerr << " " << args), ...) << std::endl;
}
#ifdef NDEBUG
#define log_debug(...) ((void)0)
#else
template <typename... Args>
void log_debug(Args... args) {
    log_info(args...);
}
#endif

using namespace Microsoft::WRL;

namespace {
struct RenderEnvironment {
    ComPtr<ID3D12Device> device;
    ComPtr<ID3D12CommandQueue> queue;
    ComPtr<IDXGISwapChain> swapChain;
};

void WaitForFence(ID3D12Fence* fence, UINT64 completionValue, HANDLE waitEvent) {
    if (fence->GetCompletedValue() < completionValue) {
        fence->SetEventOnCompletion(completionValue, waitEvent);
        WaitForSingleObject(waitEvent, INFINITE);
    }
}

RenderEnvironment CreateDeviceAndSwapChainHelper(_In_opt_ IDXGIAdapter* adapter, D3D_FEATURE_LEVEL minimumFeatureLevel,
                                                 _In_ const DXGI_SWAP_CHAIN_DESC* swapChainDesc) {
    RenderEnvironment result;

    ComPtr<IDXGIFactory4> dxgiFactory;
    auto hr = CreateDXGIFactory1(IID_PPV_ARGS(&dxgiFactory));
    ComPtr<IDXGIAdapter1> dxgiAdapter;
    if (FAILED(hr)) {
        throw std::runtime_error("DXGI factory creation failed.");
    }

    SIZE_T maxDedicatedVideoMemory = 0;
    for (UINT i = 0; dxgiFactory->EnumAdapters1(i, &dxgiAdapter) != DXGI_ERROR_NOT_FOUND; ++i) {
        DXGI_ADAPTER_DESC1 dxgiAdapterDesc;
        dxgiAdapter->GetDesc1(&dxgiAdapterDesc);

        // let's try to pickup the discrete gpu (filtering by dedicated video memory that gpu will be favored)
        if ((dxgiAdapterDesc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) == 0 &&
            SUCCEEDED(D3D12CreateDevice(dxgiAdapter.Get(), minimumFeatureLevel, __uuidof(ID3D12Device), nullptr)) &&
            dxgiAdapterDesc.DedicatedVideoMemory > maxDedicatedVideoMemory) {
            maxDedicatedVideoMemory = dxgiAdapterDesc.DedicatedVideoMemory;
        }
    }

    hr = D3D12CreateDevice(adapter, minimumFeatureLevel, IID_PPV_ARGS(&result.device));

    if (FAILED(hr)) {
        throw std::runtime_error("Device creation failed.");
    }

    D3D12_COMMAND_QUEUE_DESC queueDesc = {};
    queueDesc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
    queueDesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;

    hr = result.device->CreateCommandQueue(&queueDesc, IID_PPV_ARGS(&result.queue));

    if (FAILED(hr)) {
        throw std::runtime_error("Command queue creation failed.");
    }

    // Must copy into non-const space
    DXGI_SWAP_CHAIN_DESC swapChainDescCopy = *swapChainDesc;
    hr = dxgiFactory->CreateSwapChain(result.queue.Get(), &swapChainDescCopy, &result.swapChain);

    if (FAILED(hr)) {
        throw std::runtime_error("Swap chain creation failed.");
    }

    return result;
}
}  // namespace

DirectXRenderer::DirectXRenderer() : mWindow{nullptr, nullptr} {
}

DirectXRenderer::~DirectXRenderer() {
    Shutdown();
}

void DirectXRenderer::Render() {
    // waiting for completion of frame processing on gpu
    WaitForFence(mFrameFences[m_currentFrame].Get(), mFenceValues[m_currentFrame], mFrameFenceEvents[m_currentFrame]);

    mCommandAllocators[m_currentFrame]->Reset();

    auto commandList = mCommandLists[m_currentFrame].Get();
    commandList->Reset(mCommandAllocators[m_currentFrame].Get(), nullptr);

    D3D12_CPU_DESCRIPTOR_HANDLE renderTargetHandle;
    CD3DX12_CPU_DESCRIPTOR_HANDLE::InitOffsetted(renderTargetHandle,
                                                 mRenderTargetDescriptorHeap->GetCPUDescriptorHandleForHeapStart(),
                                                 m_currentFrame, mRenderTargetViewDescriptorSize);

    commandList->OMSetRenderTargets(1, &renderTargetHandle, true, nullptr);
    commandList->RSSetViewports(1, &mViewport);
    commandList->RSSetScissorRects(1, &mRectScissor);

    D3D12_RESOURCE_BARRIER barrierBefore;
    barrierBefore.Transition.pResource = mRenderTargets[m_currentFrame].Get();
    barrierBefore.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrierBefore.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
    barrierBefore.Transition.StateBefore = D3D12_RESOURCE_STATE_PRESENT;
    barrierBefore.Transition.StateAfter = D3D12_RESOURCE_STATE_RENDER_TARGET;
    barrierBefore.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;

    commandList->ResourceBarrier(1, &barrierBefore);

    UpdateConstantBuffer();

    static const float clearColor[] = {1.0f, 1.0f, 1.0f, 1.0f};

    commandList->ClearRenderTargetView(renderTargetHandle, clearColor, 0, nullptr);

    commandList->SetPipelineState(mPso.Get());
    commandList->SetGraphicsRootSignature(mRootSignature.Get());
    commandList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    // Set the descriptor heap containing the texture srv
    ID3D12DescriptorHeap* heaps[] = {mSrvDescriptorHeap.Get()};
    commandList->SetDescriptorHeaps(1, heaps);
    // Set slot 0 of our root signature to point to our descriptor heap with
    // the texture SRV
    commandList->SetGraphicsRootDescriptorTable(0, mSrvDescriptorHeap->GetGPUDescriptorHandleForHeapStart());
    // Set slot 1 of our root signature to the constant buffer view
    commandList->SetGraphicsRootConstantBufferView(1, mConstantBuffers[m_currentFrame]->GetGPUVirtualAddress());
    commandList->IASetVertexBuffers(0, 1, &mVertexBufferView);
    commandList->IASetIndexBuffer(&mIndexBufferView);
    commandList->DrawIndexedInstanced(6, 1, 0, 0, 0);

    D3D12_RESOURCE_BARRIER barrierAfter;
    barrierAfter.Transition.pResource = mRenderTargets[m_currentFrame].Get();
    barrierAfter.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrierAfter.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
    barrierAfter.Transition.StateBefore = D3D12_RESOURCE_STATE_RENDER_TARGET;
    barrierAfter.Transition.StateAfter = D3D12_RESOURCE_STATE_PRESENT;
    barrierAfter.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;

    commandList->ResourceBarrier(1, &barrierAfter);

    commandList->Close();

    ID3D12CommandList* commandLists[] = {commandList};
    mCommandQueue->ExecuteCommandLists(std::extent<decltype(commandLists)>::value, commandLists);

    mSwapChain->Present(CheckTearingSupport() ? 0 : 1, 0);

    // the value the gpu will set when preseting finished
    const auto fenceValue = mCurrentFenceValue;
    mCommandQueue->Signal(mFrameFences[m_currentFrame].Get(), fenceValue);
    mFenceValues[m_currentFrame] = fenceValue;
    ++mCurrentFenceValue;

    m_currentFrame = (m_currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
    assert(m_currentFrame < MAX_FRAMES_IN_FLIGHT);
}

bool DirectXRenderer::Run() {
    MSG msg;
    // regular events loop to get window responsive
    // checking msg in the window queue
    if (PeekMessage(&msg, NULL, 0, 0, PM_REMOVE)) {
        /* handle or dispatch messages */
        if (msg.message == WM_QUIT) {
            return false;
        } else {
            TranslateMessage(&msg);
            DispatchMessage(&msg);
        }
    }

    auto kb = mKeyboard->GetState();
    if (kb.Escape) {
        return false;
    }
    if (kb.W || kb.Up) {
        log_debug("Up");
    }
    if (kb.A || kb.Left) {
        log_debug("Left");
    }
    if (kb.S || kb.Down) {
        log_debug("Down");
    }
    if (kb.D || kb.Right) {
        log_debug("Right");
    }

    log_debug("x", mMouse->GetState().x, "y", mMouse->GetState().y);

    Render();

    return true;
}

/**
Setup all render targets. This creates the render target descriptor heap and
render target views for all render targets.
This function does not use a default view but instead changes the format to
_SRGB.
*/
void DirectXRenderer::SetupRenderTargets() {
    D3D12_DESCRIPTOR_HEAP_DESC heapDesc = {};
    heapDesc.NumDescriptors = MAX_FRAMES_IN_FLIGHT;
    heapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
    heapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
    mDevice->CreateDescriptorHeap(&heapDesc, IID_PPV_ARGS(&mRenderTargetDescriptorHeap));

    CD3DX12_CPU_DESCRIPTOR_HANDLE rtvHandle{mRenderTargetDescriptorHeap->GetCPUDescriptorHandleForHeapStart()};

    for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
        D3D12_RENDER_TARGET_VIEW_DESC viewDesc;
        viewDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM_SRGB;
        viewDesc.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2D;
        viewDesc.Texture2D.MipSlice = 0;
        viewDesc.Texture2D.PlaneSlice = 0;

        mDevice->CreateRenderTargetView(mRenderTargets[i].Get(), &viewDesc, rtvHandle);

        rtvHandle.Offset(mRenderTargetViewDescriptorSize);
    }
}

/**
Set up swap chain related resources, that is, the render target view, the
fences, and the descriptor heap for the render target.
*/
void DirectXRenderer::SetupSwapChain() {
    mCurrentFenceValue = 1;

    // Create fences for each frame so we can protect resources and wait for
    // any given frame
    for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
        mFrameFenceEvents[i] = CreateEvent(nullptr, FALSE, FALSE, nullptr);
        mFenceValues[i] = 0;
        mDevice->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&mFrameFences[i]));
    }

    for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
        mSwapChain->GetBuffer(i, IID_PPV_ARGS(&mRenderTargets[i]));
    }

    SetupRenderTargets();
}

void DirectXRenderer::Initialize(const std::string& title, int width, int height) {
    mWindow.reset(new Window("DirectXMiniGame", 1280, 720));
    mWindow.get_deleter() = [](Window* ptr) { delete ptr; };

    mKeyboard = std::make_unique<DirectX::Keyboard>();
    mMouse = std::make_unique<DirectX::Mouse>();
    mMouse->SetWindow(mWindow->hwnd());

    CreateDeviceAndSwapChain();

    mRectScissor = {0, 0, (long)mWindow->width(), (long)mWindow->height()};
    mViewport = {0.0f, 0.0f, static_cast<float>(mWindow->width()), static_cast<float>(mWindow->height()), 0.0f, 1.0f};

    for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
        mDevice->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&mCommandAllocators[i]));
        mDevice->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, mCommandAllocators[i].Get(), nullptr,
                                   IID_PPV_ARGS(&mCommandLists[i]));
        mCommandLists[i]->Close();
    }
    // Create our upload fence, command list and command allocator
    // This will be only used while creating the mesh buffer and the texture
    // to upload data to the GPU.
    ComPtr<ID3D12Fence> uploadFence;
    mDevice->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&uploadFence));

    ComPtr<ID3D12CommandAllocator> uploadCommandAllocator;
    mDevice->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&uploadCommandAllocator));
    ComPtr<ID3D12GraphicsCommandList> uploadCommandList;
    mDevice->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, uploadCommandAllocator.Get(), nullptr,
                               IID_PPV_ARGS(&uploadCommandList));

    // We need one descriptor heap to store our texture SRV which cannot go
    // into the root signature. So create a SRV type heap with one entry
    D3D12_DESCRIPTOR_HEAP_DESC descriptorHeapDesc = {};
    descriptorHeapDesc.NumDescriptors = 1;
    // This heap contains SRV, UAV or CBVs -- in our case one SRV
    descriptorHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    descriptorHeapDesc.NodeMask = 0;
    descriptorHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;

    mDevice->CreateDescriptorHeap(&descriptorHeapDesc, IID_PPV_ARGS(&mSrvDescriptorHeap));
    CreateRootSignature();
    CreatePipelineStateObject();
    CreateConstantBuffer();
    CreateTexture(uploadCommandList.Get());
    CreateMeshBuffers(uploadCommandList.Get());

    uploadCommandList->Close();

    // Execute the upload and finish the command list
    ID3D12CommandList* commandLists[] = {uploadCommandList.Get()};
    mCommandQueue->ExecuteCommandLists(std::extent<decltype(commandLists)>::value, commandLists);
    mCommandQueue->Signal(uploadFence.Get(), 1);

    auto waitEvent = CreateEvent(nullptr, FALSE, FALSE, nullptr);

    if (waitEvent == NULL) {
        throw std::runtime_error("Could not create wait event.");
    }

    WaitForFence(uploadFence.Get(), 1, waitEvent);

    // Cleanup our upload handle
    uploadCommandAllocator->Reset();

    CloseHandle(waitEvent);
}

void DirectXRenderer::CreateMeshBuffers(ID3D12GraphicsCommandList* uploadCommandList) {
    struct Vertex {
        float position[3];
        float uv[2];
    };

    // Declare upload buffer data as 'static' so it persists after returning from this function.
    // Otherwise, we would need to explicitly wait for the GPU to copy data from the upload buffer
    // to vertex/index default buffers due to how the GPU processes commands asynchronously.
    static const Vertex vertices[4] = {// Upper Left
                                       {{-1.0f, 1.0f, 0}, {0, 0}},
                                       // Upper Right
                                       {{1.0f, 1.0f, 0}, {1, 0}},
                                       // Bottom right
                                       {{1.0f, -1.0f, 0}, {1, 1}},
                                       // Bottom left
                                       {{-1.0f, -1.0f, 0}, {0, 1}}};

    static const int indices[6] = {0, 1, 2, 2, 3, 0};

    static const int uploadBufferSize = sizeof(vertices) + sizeof(indices);
    static const auto uploadHeapProperties = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD);
    static const auto uploadBufferDesc = CD3DX12_RESOURCE_DESC::Buffer(uploadBufferSize);

    // Create upload buffer on CPU
    mDevice->CreateCommittedResource(&uploadHeapProperties, D3D12_HEAP_FLAG_NONE, &uploadBufferDesc,
                                     D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&mUploadBuffer));

    // Create vertex & index buffer on the GPU
    // HEAP_TYPE_DEFAULT is on GPU, we also initialize with COPY_DEST state
    // so we don't have to transition into this before copying into them
    static const auto defaultHeapProperties = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT);

    static const auto vertexBufferDesc = CD3DX12_RESOURCE_DESC::Buffer(sizeof(vertices));
    mDevice->CreateCommittedResource(&defaultHeapProperties, D3D12_HEAP_FLAG_NONE, &vertexBufferDesc,
                                     D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&mVertexBuffer));

    static const auto indexBufferDesc = CD3DX12_RESOURCE_DESC::Buffer(sizeof(indices));
    mDevice->CreateCommittedResource(&defaultHeapProperties, D3D12_HEAP_FLAG_NONE, &indexBufferDesc,
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

void DirectXRenderer::CreatePipelineStateObject() {
    static const D3D12_INPUT_ELEMENT_DESC layout[] = {
        {"POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
        {"TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT, 0, 12, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0}};

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
    psoDesc.DSVFormat = DXGI_FORMAT_UNKNOWN;
    psoDesc.InputLayout.NumElements = std::extent<decltype(layout)>::value;
    psoDesc.InputLayout.pInputElementDescs = layout;
    psoDesc.RasterizerState = CD3DX12_RASTERIZER_DESC(D3D12_DEFAULT);
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
    psoDesc.DepthStencilState.DepthEnable = false;
    psoDesc.DepthStencilState.StencilEnable = false;
    psoDesc.SampleMask = 0xFFFFFFFF;
    psoDesc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;

    mDevice->CreateGraphicsPipelineState(&psoDesc, IID_PPV_ARGS(&mPso));
}

void DirectXRenderer::CreateTextureWIC() {
    DirectX::ResourceUploadBatch resourceUpload(mDevice.Get());
    resourceUpload.Begin();
    DirectX::CreateWICTextureFromFileEx(mDevice.Get(), resourceUpload, L"textures\\texture.png", 0, D3D12_RESOURCE_FLAG_NONE,
                                        DirectX::WIC_LOADER_FORCE_RGBA32 | DirectX::WIC_LOADER_MIP_AUTOGEN,
                                        mImage.ReleaseAndGetAddressOf());
    // Upload the resources to the GPU.
    auto uploadResourcesFinished = resourceUpload.End(mCommandQueue.Get());
    // Wait for the upload thread to terminate
    uploadResourcesFinished.wait();
}

void DirectXRenderer::CreateTexture(ID3D12GraphicsCommandList* uploadCommandList) {
    int texWidth, texHeight, texChannels;
    std::size_t imageSizeTotal = 0u;
    using dataTexturetPtr = std::unique_ptr<stbi_uc, decltype(&stbi_image_free)>;

    std::string path = "textures\\texture.png";

    /// STBI_rgb_alpha coerces to have ALPHA chanel for consistency with alphaless images
    stbi_uc* pixels = stbi_load(path.c_str(), &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);
    assert(pixels);
    dataTexturetPtr textureData(pixels, stbi_image_free);
    imageSizeTotal += texWidth * texHeight * 4LL;

    static const auto defaultHeapProperties = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT);
    const auto resourceDesc = CD3DX12_RESOURCE_DESC::Tex2D(DXGI_FORMAT_R8G8B8A8_UNORM_SRGB, texWidth, texHeight, 1, 1);

    mDevice->CreateCommittedResource(&defaultHeapProperties, D3D12_HEAP_FLAG_NONE, &resourceDesc, D3D12_RESOURCE_STATE_COPY_DEST,
                                     nullptr, IID_PPV_ARGS(&mImage));

    static const auto uploadHeapProperties = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD);
    const auto uploadBufferSize = GetRequiredIntermediateSize(mImage.Get(), 0, 1);
    const auto uploadBufferDesc = CD3DX12_RESOURCE_DESC::Buffer(uploadBufferSize);

    mDevice->CreateCommittedResource(&uploadHeapProperties, D3D12_HEAP_FLAG_NONE, &uploadBufferDesc,
                                     D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&mUploadImage));

    D3D12_SUBRESOURCE_DATA srcData;
    srcData.pData = textureData.get();
    srcData.RowPitch = texWidth * 4;
    srcData.SlicePitch = texWidth * texHeight * 4;

    UpdateSubresources(uploadCommandList, mImage.Get(), mUploadImage.Get(), 0, 0, 1, &srcData);
    const auto transition = CD3DX12_RESOURCE_BARRIER::Transition(mImage.Get(), D3D12_RESOURCE_STATE_COPY_DEST,
                                                                 D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
    uploadCommandList->ResourceBarrier(1, &transition);

    D3D12_SHADER_RESOURCE_VIEW_DESC shaderResourceViewDesc = {};
    shaderResourceViewDesc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    shaderResourceViewDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    shaderResourceViewDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM_SRGB;
    shaderResourceViewDesc.Texture2D.MipLevels = 1;
    shaderResourceViewDesc.Texture2D.MostDetailedMip = 0;
    shaderResourceViewDesc.Texture2D.ResourceMinLODClamp = 0.0f;

    mDevice->CreateShaderResourceView(mImage.Get(), &shaderResourceViewDesc,
                                      mSrvDescriptorHeap->GetCPUDescriptorHandleForHeapStart());
}

void DirectXRenderer::CreateRootSignature() {
    // We have two root parameters, one is a pointer to a descriptor heap
    // with a SRV, the second is a constant buffer view
    CD3DX12_ROOT_PARAMETER parameters[2];

    // Create a descriptor table with one entry in our descriptor heap
    CD3DX12_DESCRIPTOR_RANGE range{D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0};
    parameters[0].InitAsDescriptorTable(1, &range);

    // Our constant buffer view
    parameters[1].InitAsConstantBufferView(0, 0, D3D12_SHADER_VISIBILITY_VERTEX);

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

    mDevice->CreateRootSignature(0, rootBlob->GetBufferPointer(), rootBlob->GetBufferSize(), IID_PPV_ARGS(&mRootSignature));
}

void DirectXRenderer::CreateConstantBuffer() {
    for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
        static const auto uploadHeapProperties = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD);
        static const auto constantBufferDesc = CD3DX12_RESOURCE_DESC::Buffer(sizeof(ConstantBuffer));

        mDevice->CreateCommittedResource(&uploadHeapProperties, D3D12_HEAP_FLAG_NONE, &constantBufferDesc,
                                         D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&mConstantBuffers[i]));

        void* p;
        mConstantBuffers[i]->Map(0, nullptr, &p);
        memcpy(p, &mConstantBufferData, sizeof(mConstantBufferData));
        mConstantBuffers[i]->Unmap(0, nullptr);
    }
}

void DirectXRenderer::UpdateConstantBuffer() {
    static int counter = 0;
    counter++;
    mConstantBufferData.x = std::abs(std::sin(static_cast<float>(counter) / 64.0f));

    void* data;
    mConstantBuffers[m_currentFrame]->Map(0, nullptr, &data);
    memcpy(data, &mConstantBufferData, sizeof(mConstantBufferData));
    mConstantBuffers[m_currentFrame]->Unmap(0, nullptr);
}

void DirectXRenderer::Shutdown() {
    // Drain the queue, wait for everything to finish
    for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
        WaitForFence(mFrameFences[i].Get(), mFenceValues[i], mFrameFenceEvents[i]);
    }

    for (auto event : mFrameFenceEvents) {
        CloseHandle(event);
    }
}

// checking for G-SYNC or Free-Sync availability to avoid v-sync and posibble cpu blocking
bool DirectXRenderer::CheckTearingSupport() {
    BOOL allowTearing = FALSE;

    ComPtr<IDXGIFactory5> factory;
    if (SUCCEEDED(CreateDXGIFactory1(IID_PPV_ARGS(&factory)))) {
        if (FAILED(factory->CheckFeatureSupport(DXGI_FEATURE_PRESENT_ALLOW_TEARING, &allowTearing, sizeof(allowTearing)))) {
            allowTearing = FALSE;
        }
    }

    return allowTearing == TRUE;
}

void DirectXRenderer::CreateDeviceAndSwapChain() {
    // Enable the debug layers when in debug mode
    // you need get Graphics Tools installed to debug DirectX
#ifdef _DEBUG
    ComPtr<ID3D12Debug> debugController;
    D3D12GetDebugInterface(IID_PPV_ARGS(&debugController));
    debugController->EnableDebugLayer();
#endif

    DXGI_SWAP_CHAIN_DESC swapChainDesc;
    ::ZeroMemory(&swapChainDesc, sizeof(swapChainDesc));

    swapChainDesc.BufferCount = MAX_FRAMES_IN_FLIGHT;
    swapChainDesc.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    swapChainDesc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    swapChainDesc.BufferDesc.Width = mWindow->width();
    swapChainDesc.BufferDesc.Height = mWindow->height();
    swapChainDesc.OutputWindow = mWindow->hwnd();
    swapChainDesc.SampleDesc.Count = 1;
    // DXGI_SWAP_CHAIN_DESC1 supports buffer\swapChain mismatch -> it will strech buffer to fit into swapChain
    // swapChainDesc.Scaling = DXGI_SCALING_STRETCH;
    swapChainDesc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
    swapChainDesc.Windowed = true;
    swapChainDesc.Flags = CheckTearingSupport() ? DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING : 0;  // set free-sync\g-sync instead v-sync

    // the driver may support directx 12 api but without hardware acceleration
    // D3D_FEATURE_LEVEL_11_0 hardware acceleration is present for sure
    auto renderEnv = CreateDeviceAndSwapChainHelper(nullptr, D3D_FEATURE_LEVEL_11_0, &swapChainDesc);

    mDevice = renderEnv.device;
    mCommandQueue = renderEnv.queue;
    mSwapChain = renderEnv.swapChain;

    mRenderTargetViewDescriptorSize = mDevice->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);

    SetupSwapChain();
}
