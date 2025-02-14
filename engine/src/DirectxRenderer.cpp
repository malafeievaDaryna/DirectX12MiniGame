#include "DirectXRenderer.h"
#include "Window.h"
/**
 * d3dx12.h provides some useful classes that will simplify some of the functions
 * it needs to be downloaded separately from the Microsoft DirectX repository
 * (https://github.com/Microsoft/DirectX-Graphics-Samples/tree/master/Libraries/D3DX12)
 */
#include "Shaders.h"
#include "Utils.h"
#include "d3dx12.h"

#include <d3dcompiler.h>
#include <dxgi1_5.h>
#include <ResourceUploadBatch.h >
#include <algorithm>
#include <cassert>

#include <imgui/backends/imgui_impl_dx12.h>
#include <imgui/backends/imgui_impl_win32.h>
#include <imgui/imgui.h>

using namespace Microsoft::WRL;
using namespace constants;
using namespace utils;

namespace {

static constexpr float FAR_PLANE = 10000.0f;

const static DirectX::XMVECTOR _upDir = DirectX::XMVectorSet(0.0f, 1.0f, 0.0f, 0.0f);

// md5 models have different dirrection we must adjust them
static constexpr float _md5AngleAdjustment = -90.0f;
const static XMVECTOR _md5RotAdjustment = XMQuaternionRotationNormal(_upDir, XMConvertToRadians(_md5AngleAdjustment));
const static XMVECTOR _md5RotAdjustmentConj = XMQuaternionConjugate(_md5RotAdjustment);

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

    SIZE_T maxDedicatedVideoMemory = 0u;
    UINT dedicatedIndex = 0u;
    for (UINT i = 0u; dxgiFactory->EnumAdapters1(i, &dxgiAdapter) != DXGI_ERROR_NOT_FOUND; ++i) {
        DXGI_ADAPTER_DESC1 dxgiAdapterDesc;
        dxgiAdapter->GetDesc1(&dxgiAdapterDesc);

        // let's try to pickup the discrete gpu (filtering by dedicated video memory that gpu will be favored)
        if ((dxgiAdapterDesc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) == 0 &&
            SUCCEEDED(D3D12CreateDevice(dxgiAdapter.Get(), minimumFeatureLevel, __uuidof(ID3D12Device), nullptr)) &&
            dxgiAdapterDesc.DedicatedVideoMemory > maxDedicatedVideoMemory) {
            maxDedicatedVideoMemory = dxgiAdapterDesc.DedicatedVideoMemory;
            dedicatedIndex = i;
        }
    }

    dxgiFactory->EnumAdapters1(dedicatedIndex, &dxgiAdapter);

    hr = D3D12CreateDevice(dxgiAdapter.Get(), minimumFeatureLevel, IID_PPV_ARGS(&result.device));

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

void getAndUpdateMonsterPosition(DirectX::XMFLOAT3& monsterBasePos, const XMFLOAT3& animationPosShift,
                                 const DirectX::XMVECTOR& quatMonsterRot) {
    XMVECTOR animationPosShiftVec = XMLoadFloat3(&animationPosShift);
    XMVECTOR rotatedPosVec =
        XMQuaternionMultiply(XMQuaternionMultiply(quatMonsterRot, animationPosShiftVec), XMQuaternionConjugate(quatMonsterRot));
    XMFLOAT4 rotatedPos;
    XMStoreFloat4(&rotatedPos, rotatedPosVec);
    monsterBasePos.x += rotatedPos.x;
    monsterBasePos.y += rotatedPos.y;
    monsterBasePos.z += -rotatedPos.z;
}
}  // namespace

DirectXRenderer::DirectXRenderer() : mWindow{nullptr, nullptr} {
}

DirectXRenderer::~DirectXRenderer() {
    Shutdown();
}

void DirectXRenderer::Render() {
    static auto startTime = std::chrono::high_resolution_clock::now();
    static auto endTime = std::chrono::high_resolution_clock::now();

    // Start the Dear ImGui frame
    ImGui_ImplDX12_NewFrame();
    ImGui_ImplWin32_NewFrame();
    ImGui::NewFrame();
#ifdef _DEBUG
    static std::pair<float, std::pair<uint32_t, float>> FPS_METRICS{
        0.0f, {0u, 0.0f}};  // current FPS value and frames per elapsed milliseconds
    ImGui::Begin("Debug Window", nullptr, ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoResize);
    ImGui::SetNextWindowBgAlpha(0.5f);
    ImGui::SetWindowFontScale(1.3);
    ImGui::SetWindowSize(ImVec2(0, 0));
    ImGui::BeginChild("First", ImVec2(200, 30));
    ImGui::Text("FPS %.2f", FPS_METRICS.first);
    ImGui::EndChild();
    float f = 0.5;
    ImGui::SliderFloat("float", &f, 0.0f, 1.0f);
    ImGui::End();
#endif
    ImGui::Render();

    // waiting for completion of frame processing on gpu
    WaitForFence(mFrameFences[m_currentFrame].Get(), mFenceValues[m_currentFrame], mFrameFenceEvents[m_currentFrame]);

    mCommandAllocators[m_currentFrame]->Reset();

    auto commandList = mCommandLists[m_currentFrame].Get();
    commandList->Reset(mCommandAllocators[m_currentFrame].Get(), nullptr);

    // prepare RenderTargets\Depth handlers
    D3D12_CPU_DESCRIPTOR_HANDLE renderTargetHandle;
    CD3DX12_CPU_DESCRIPTOR_HANDLE::InitOffsetted(renderTargetHandle,
                                                 mRenderTargetDescriptorHeap->GetCPUDescriptorHandleForHeapStart(),
                                                 m_currentFrame, mRenderTargetViewDescriptorSize);
    CD3DX12_CPU_DESCRIPTOR_HANDLE dsvHandle(mDSDescriptorHeap->GetCPUDescriptorHandleForHeapStart());

    commandList->OMSetRenderTargets(1, &renderTargetHandle, true, &dsvHandle);
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

    assert(mPistolAnimationsActions.find(mCurPistolAnim) != mPistolAnimationsActions.end());
    md5PistolModel->UpdateMD5Model(frameTimeMS, static_cast<int>(mCurPistolAnim), mPistolAnimationsActions[mCurPistolAnim]);
    assert(mMonsterAnimationsActions.find(mCurMonsterAnim) != mMonsterAnimationsActions.end());
    md5MonsterModel->UpdateMD5Model(frameTimeMS, static_cast<int>(mCurMonsterAnim), mMonsterAnimationsActions[mCurMonsterAnim]);

    mSkyBox->Update(m_currentFrame, mCamera->viewProjMat());

    UpdateConstantBuffer();

    static const float clearColor[] = {1.0f, 1.0f, 1.0f, 1.0f};

    commandList->ClearRenderTargetView(renderTargetHandle, clearColor, 0, nullptr);
    commandList->ClearDepthStencilView(mDSDescriptorHeap->GetCPUDescriptorHandleForHeapStart(), D3D12_CLEAR_FLAG_DEPTH, 1.0f, 0,
                                       0, nullptr);

    commandList->SetPipelineState(mPso.Get());
    commandList->SetGraphicsRootSignature(mRootSignature.Get());
    commandList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    // Set slot 1 of our root signature to the constant buffer view
    commandList->SetGraphicsRootConstantBufferView(1, mPistolConstantBuffers[m_currentFrame]->GetGPUVirtualAddress());
    md5PistolModel->Draw(commandList);

    commandList->SetGraphicsRootConstantBufferView(1, mMonsterConstantBuffers[m_currentFrame]->GetGPUVirtualAddress());
    md5MonsterModel->Draw(commandList);

    // landscape
    {
        ID3D12DescriptorHeap* heaps[] = {mLandscapeTexture.srvDescriptorHeap.Get()};
        commandList->SetDescriptorHeaps(1, heaps);
        // Set slot 0 of our root signature to point to our descriptor heap with
        // the texture SRV
        commandList->SetGraphicsRootDescriptorTable(0, mLandscapeTexture.srvDescriptorHeap->GetGPUDescriptorHandleForHeapStart());

        commandList->SetGraphicsRootConstantBufferView(1, mLandScapeConstantBuffers[m_currentFrame]->GetGPUVirtualAddress());

        commandList->IASetVertexBuffers(0, 1, &mVertexBufferView);
        commandList->IASetIndexBuffer(&mIndexBufferView);
        commandList->DrawIndexedInstanced(6, 1, 0, 0, 0);
    }

    mSkyBox->Draw(m_currentFrame, commandList);

    {
        ID3D12DescriptorHeap* heaps[] = {mImGuiDescriptorHeap.Get()};
        commandList->SetDescriptorHeaps(1, heaps);
        ImGui_ImplDX12_RenderDrawData(ImGui::GetDrawData(), commandList);
    }

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

    mSwapChain->Present(mIsTearingSupport ? 0 : 1, 0);

    // the value the gpu will set when preseting finished
    const auto fenceValue = mCurrentFenceValue;
    mCommandQueue->Signal(mFrameFences[m_currentFrame].Get(), fenceValue);
    mFenceValues[m_currentFrame] = fenceValue;
    ++mCurrentFenceValue;

    m_currentFrame = (m_currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
    assert(m_currentFrame < MAX_FRAMES_IN_FLIGHT);

    endTime = std::chrono::high_resolution_clock::now();
    frameTimeMS = std::chrono::duration<float, std::chrono::milliseconds::period>(endTime - startTime).count();
    startTime = endTime;

#ifdef _DEBUG
    FPS_METRICS.second.first += 1;             // plus one frame
    FPS_METRICS.second.second += frameTimeMS;  // plus the time spent to draw the frame
    if (FPS_METRICS.second.second >= 1000u) {  // how many frames have we drawn within ~1sec
        FPS_METRICS.first = FPS_METRICS.second.second * FPS_METRICS.second.first / 1000.0f;
        FPS_METRICS.second = std::pair<uint32_t, float>{0u, 0.0f};  // actual FPS
    }
#endif
}

bool DirectXRenderer::Run() {
    // log_debug("frameTimeMS ", frameTimeMS);
    /** For debug only to lock FPS
    float sleep = constants::_144_FPS_TO_MS - frameTimeMS;
    sleep = sleep > 0 ? sleep : 0;
    Sleep(sleep);
    */

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

    if (mMouse->GetState().leftButton) {
        mCurPistolAnim = PISTOL_ANIM::FIRE;
    }

    auto kb = mKeyboard->GetState();

    POINT mousePos;
    GetCursorPos(&mousePos);
    // log_debug("mousePos x", mousePos.x, " mousePos y ", mousePos.y);

    static int lastMouseX = mousePos.x;
    static int lastMouseY = mousePos.y;

    int deltaMouseX = mousePos.x - lastMouseX;

    const auto defaultMousePos = mWindow->resetMousePos();
    lastMouseX = defaultMousePos.x;
    lastMouseY = defaultMousePos.y;

    if (deltaMouseX < 0) {
        mCamera->update(frameTimeMS, Camera::EDirection::Turn_Left);
    } else if (deltaMouseX > 0) {
        mCamera->update(frameTimeMS, Camera::EDirection::Turn_Right);
    }

    if (kb.Escape) {
        return false;
    }
    if (kb.W || kb.Up) {
        mCamera->update(frameTimeMS, Camera::EDirection::Forward);
    }
    if (kb.A || kb.Left) {
        mCamera->update(frameTimeMS, Camera::EDirection::Left);
    }
    if (kb.S || kb.Down) {
        mCamera->update(frameTimeMS, Camera::EDirection::Back);
    }
    if (kb.D || kb.Right) {
        mCamera->update(frameTimeMS, Camera::EDirection::Right);
    }

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
    if (FAILED(mDevice->CreateDescriptorHeap(&heapDesc, IID_PPV_ARGS(&mRenderTargetDescriptorHeap)))) {
        log_err("Couldn't allocate gpu heap memory");
    }

    CD3DX12_CPU_DESCRIPTOR_HANDLE rtvHandle{mRenderTargetDescriptorHeap->GetCPUDescriptorHandleForHeapStart()};

    for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
        D3D12_RENDER_TARGET_VIEW_DESC viewDesc;
        viewDesc.Format = constants::RENDER_TARGET_FORMAT;
        viewDesc.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2D;
        viewDesc.Texture2D.MipSlice = 0;
        viewDesc.Texture2D.PlaneSlice = 0;

        mDevice->CreateRenderTargetView(mRenderTargets[i].Get(), &viewDesc, rtvHandle);

        rtvHandle.Offset(mRenderTargetViewDescriptorSize);
    }

    // create a depth stencil descriptor heap
    D3D12_DESCRIPTOR_HEAP_DESC dsvHeapDesc = {};
    dsvHeapDesc.NumDescriptors = 1;
    dsvHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_DSV;
    dsvHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
    if (FAILED(mDevice->CreateDescriptorHeap(&dsvHeapDesc, IID_PPV_ARGS(&mDSDescriptorHeap)))) {
        log_err("Couldn't allocate gpu heap memory");
    }

    D3D12_DEPTH_STENCIL_VIEW_DESC depthStencilDesc = {};
    depthStencilDesc.Format = DXGI_FORMAT_D32_FLOAT;
    depthStencilDesc.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2D;
    depthStencilDesc.Flags = D3D12_DSV_FLAG_NONE;

    D3D12_CLEAR_VALUE depthOptimizedClearValue = {};
    depthOptimizedClearValue.Format = DEPTH_FORMAT;
    depthOptimizedClearValue.DepthStencil.Depth = 1.0f;
    depthOptimizedClearValue.DepthStencil.Stencil = 0;

    mDevice->CreateCommittedResource(&CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT), D3D12_HEAP_FLAG_NONE,
                                     &CD3DX12_RESOURCE_DESC::Tex2D(DEPTH_FORMAT, mWindow->width(), mWindow->height(), 1, 0, 1, 0,
                                                                   D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL),
                                     D3D12_RESOURCE_STATE_DEPTH_WRITE, &depthOptimizedClearValue,
                                     IID_PPV_ARGS(&mDepthStencilBuffer));
    mDSDescriptorHeap->SetName(L"Depth/Stencil Resource Heap");

    mDevice->CreateDepthStencilView(mDepthStencilBuffer.Get(), &depthStencilDesc,
                                    mDSDescriptorHeap->GetCPUDescriptorHandleForHeapStart());
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
    // Check for DirectX Math library support.
    if (!DirectX::XMVerifyCPUSupport()) {
        MessageBoxA(NULL, "Failed to verify DirectX Math library support.", "Error", MB_OK | MB_ICONERROR);
        std::exit(-1);
    }

    mWindow.reset(new Window("DirectXMiniGame", 1920, 1080));
    mWindow.get_deleter() = [](Window* ptr) { delete ptr; };

    mKeyboard = std::make_unique<DirectX::Keyboard>();
    mMouse = std::make_unique<DirectX::Mouse>();

    Camera::Perstective perspective;
    float aspectRatio = static_cast<float>(mWindow->width()) / mWindow->height();
    perspective.fovy = 45.0f;
    perspective.aspect = aspectRatio;
    perspective._near = 0.01f;
    perspective._far = FAR_PLANE;
    mCamera = std::make_unique<Camera>(perspective, DirectX::XMFLOAT4{0, 50, -1800, 0}, DirectX::XMFLOAT4{0, 50, 0, 0});

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
    CreateRootSignature();
    CreatePipelineStateObject();

    CreateConstantBuffer();

    // landscape
    mLandscapeTexture = utils::CreateTexture(mDevice.Get(), uploadCommandList.Get(), "landscape.png");
    CreateMeshBuffers(uploadCommandList.Get());

    mPistolAnimationsActions[PISTOL_ANIM::IDLE] = nullptr;
    mPistolAnimationsActions[PISTOL_ANIM::RELOAD] = [this]() { mCurPistolAnim = PISTOL_ANIM::IDLE; };
    mPistolAnimationsActions[PISTOL_ANIM::FIRE] = [this]() {
        mCurPistolAnim = PISTOL_ANIM::IDLE;

        DirectX::XMFLOAT3 monsterBasePosTempt = mMonsterBasePos;
        getAndUpdateMonsterPosition(monsterBasePosTempt, md5MonsterModel->GetPosDiffFirstLastFrames(), mQuatMonsterRot);

        const XMFLOAT4& eye = mCamera->cameraPosition();
        XMVECTOR eyeVec = XMLoadFloat4(&eye);
        XMVECTOR monsterPosVec = XMLoadFloat3(&monsterBasePosTempt);
        XMVECTOR fromCameraToMonsterLengthVec = XMVector3Length(XMVectorSubtract(monsterPosVec, eyeVec));
        XMFLOAT4 length;
        XMStoreFloat4(&length, fromCameraToMonsterLengthVec);

        const XMFLOAT4& eyeDir = mCamera->targetPosition();
        XMVECTOR eyeDirVec = XMLoadFloat4(&eyeDir);

        XMVECTOR bulletDestPosVec = XMVectorAdd(eyeVec, XMVectorScale(eyeDirVec, length.x));
        XMVECTOR fromBulletToMonsterLengthVec = XMVector3Length(XMVectorSubtract(monsterPosVec, bulletDestPosVec));
        XMStoreFloat4(&length, fromBulletToMonsterLengthVec);
        // log_debug("fromBulletToMonsterLengthVec ", length.x);
        // log_debug("radius ", md5MonsterModel->GetRadius());

        if (md5MonsterModel->GetRadius() >= length.x) {
            mCurMonsterAnim = MONSTER_ANIM::PAIN;
        }
    };

    std::unordered_map<PISTOL_ANIM, std::string> pistolAnimations = {{PISTOL_ANIM::IDLE, "models/pistol_idle.md5anim"},
                                                                     {PISTOL_ANIM::RELOAD, "models/pistol_reload.md5anim"},
                                                                     {PISTOL_ANIM::FIRE, "models/pistol_fire.md5anim"}};
    md5PistolModel = std::make_unique<MD5Loader>(
        mDevice.Get(), uploadCommandList.Get(), "models/pistol.md5mesh",
        std::vector<std::string>{pistolAnimations[PISTOL_ANIM::IDLE], pistolAnimations[PISTOL_ANIM::RELOAD],
                                 pistolAnimations[PISTOL_ANIM::FIRE]});

    mMonsterAnimationsActions[MONSTER_ANIM::IDLE] = nullptr;
    mMonsterAnimationsActions[MONSTER_ANIM::PAIN] = [this]() { mCurMonsterAnim = MONSTER_ANIM::IDLE; };
    mMonsterAnimationsActions[MONSTER_ANIM::RUN] = [this]() {
        getAndUpdateMonsterPosition(mMonsterBasePos, md5MonsterModel->GetPosDiffFirstLastFrames(), mQuatMonsterRot);
        // log_debug("mMonsterBasePos ", mMonsterBasePos.x, " ", mMonsterBasePos.y, " ", mMonsterBasePos.z);

        const XMFLOAT4& eye = mCamera->cameraPosition();
        // eye.y = 0; any way we ignore Y when getting 'angle' below
        // log_debug("mCamera ", eye.x, " ", eye.y, " ", eye.z);
        XMVECTOR eyeVec = XMLoadFloat4(&eye);
        XMVECTOR monsterPosVec = XMLoadFloat3(&mMonsterBasePos);
        XMVECTOR fromMonsterToCameraVec = XMVector3Normalize(XMVectorSubtract(eyeVec, monsterPosVec));
        XMFLOAT4 fromMonsterToCameraPos;
        XMStoreFloat4(&fromMonsterToCameraPos, fromMonsterToCameraVec);

        /**  Note: we can use faster algorithm for monster rotation calculation than XMMatrixLookAtLH
        auto tempQuat =
            XMQuaternionRotationMatrix(XMMatrixLookAtLH(monsterPosVec, XMVectorAdd(monsterPosVec, fromMonsterToCamera), _upDir));
            */
        XMFLOAT4 newQuat;
        float angle = atan2f(fromMonsterToCameraPos.x, fromMonsterToCameraPos.z);
        newQuat.x = 0;
        newQuat.y = 1.0f * sin(angle / 2.0f);
        newQuat.z = 0;
        newQuat.w = -cos(angle / 2.0f);
        XMVECTOR newQuatVec = XMLoadFloat4(&newQuat);

        newQuatVec = XMQuaternionMultiply(_md5RotAdjustment,
                                          newQuatVec);  // models directed along x axis, let's align them to positive z axis
        // we need rotate the monster instead rotating the scene around the monster
        mQuatMonsterRot = XMQuaternionInverse(newQuatVec);
        mQuatMonsterRot = XMQuaternionMultiply(mQuatMonsterRot, XMQuaternionRotationNormal(_upDir, XMConvertToRadians(180.0f)));
    };
    std::unordered_map<MONSTER_ANIM, std::string> monsterAnimations = {{MONSTER_ANIM::IDLE, "models/pinky_stand.md5anim"},
                                                                       {MONSTER_ANIM::RUN, "models/pinky_run.md5anim"},
                                                                       {MONSTER_ANIM::PAIN, "models/pinky_pain.md5anim"}};
    md5MonsterModel = std::make_unique<MD5Loader>(
        mDevice.Get(), uploadCommandList.Get(), "models/pinky.md5mesh",
        std::vector<std::string>{monsterAnimations[MONSTER_ANIM::IDLE], monsterAnimations[MONSTER_ANIM::RUN],
                                 monsterAnimations[MONSTER_ANIM::PAIN]});

    mSkyBox = std::make_unique<SkyBox>(mDevice.Get(), mCommandQueue.Get(), uploadCommandList.Get(), "skybox.dds");

    SetupImGui();

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

void DirectXRenderer::SetupImGui() {
    D3D12_DESCRIPTOR_HEAP_DESC heapDesc = {};
    heapDesc.NumDescriptors = 100;  // 100 descriptors should be enough
    heapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    heapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    if (FAILED(mDevice->CreateDescriptorHeap(&heapDesc, IID_PPV_ARGS(&mImGuiDescriptorHeap)))) {
        log_err("Couldn't allocate gpu heap memory");
    }

    static utils::DescriptorHeapAllocator _ImGuiDescriptorAlloc(mDevice.Get(), mImGuiDescriptorHeap.Get());

    ImGui_ImplDX12_InitInfo init_info = {};
    init_info.Device = mDevice.Get();
    init_info.CommandQueue = mCommandQueue.Get();
    init_info.NumFramesInFlight = constants::MAX_FRAMES_IN_FLIGHT;
    init_info.RTVFormat = constants::RENDER_TARGET_FORMAT;
    init_info.DSVFormat = constants::DEPTH_FORMAT;
    // we provide callbacks for allocating SRV descriptors (for textures)
    init_info.SrvDescriptorHeap = mImGuiDescriptorHeap.Get();
    init_info.SrvDescriptorAllocFn = [](ImGui_ImplDX12_InitInfo*, D3D12_CPU_DESCRIPTOR_HANDLE* out_cpu_handle,
                                        D3D12_GPU_DESCRIPTOR_HANDLE* out_gpu_handle) {
        return _ImGuiDescriptorAlloc.Alloc(out_cpu_handle, out_gpu_handle);
    };
    init_info.SrvDescriptorFreeFn = [](ImGui_ImplDX12_InitInfo*, D3D12_CPU_DESCRIPTOR_HANDLE cpu_handle,
                                       D3D12_GPU_DESCRIPTOR_HANDLE gpu_handle) {
        return _ImGuiDescriptorAlloc.Free(cpu_handle, gpu_handle);
    };
    ImGui_ImplDX12_Init(&init_info);
}

void DirectXRenderer::CreateMeshBuffers(ID3D12GraphicsCommandList* uploadCommandList) {
    struct Vertex {
        float position[3];
        float uv[2];
    };

    constexpr float landscapeScaleFactor = FAR_PLANE;
    const Vertex vertices[4] = {{{-1.0f * landscapeScaleFactor, 0.0f, -1.0f * landscapeScaleFactor}, {0, 0}},
                                {{-1.0f * landscapeScaleFactor, 0.0f, 1.0f * landscapeScaleFactor}, {1, 0}},
                                {{1.0f * landscapeScaleFactor, 0.0f, 1.0f * landscapeScaleFactor}, {1, 1}},
                                {{1.0f * landscapeScaleFactor, 0.0f, -1.0f * landscapeScaleFactor}, {0, 1}}};

    const int indices[6] = {0, 1, 2, 2, 3, 0};

    const int uploadBufferSize = sizeof(vertices) + sizeof(indices);
    const auto uploadHeapProperties = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD);
    const auto uploadBufferDesc = CD3DX12_RESOURCE_DESC::Buffer(uploadBufferSize);

    // Create upload buffer on CPU
    mDevice->CreateCommittedResource(&uploadHeapProperties, D3D12_HEAP_FLAG_NONE, &uploadBufferDesc,
                                     D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&mUploadBuffer));

    // Create vertex & index buffer on the GPU
    // HEAP_TYPE_DEFAULT is on GPU, we also initialize with COPY_DEST state
    // so we don't have to transition into this before copying into them
    const auto defaultHeapProperties = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT);

    const auto vertexBufferDesc = CD3DX12_RESOURCE_DESC::Buffer(sizeof(vertices));
    mDevice->CreateCommittedResource(&defaultHeapProperties, D3D12_HEAP_FLAG_NONE, &vertexBufferDesc,
                                     D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&mVertexBuffer));

    const auto indexBufferDesc = CD3DX12_RESOURCE_DESC::Buffer(sizeof(indices));
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
    D3DCompile(shaders::vs_shader, sizeof(shaders::vs_shader), "", nullptr, nullptr, "VS_main", "vs_5_1", compileFlags, 0,
               &vertexShader, nullptr);

    ComPtr<ID3DBlob> pixelShader;
    D3DCompile(shaders::fs_shader, sizeof(shaders::fs_shader), "", nullptr, nullptr, "PS_main", "ps_5_1", compileFlags, 0,
               &pixelShader, nullptr);

    D3D12_GRAPHICS_PIPELINE_STATE_DESC psoDesc = {};
    psoDesc.VS.BytecodeLength = vertexShader->GetBufferSize();
    psoDesc.VS.pShaderBytecode = vertexShader->GetBufferPointer();
    psoDesc.PS.BytecodeLength = pixelShader->GetBufferSize();
    psoDesc.PS.pShaderBytecode = pixelShader->GetBufferPointer();
    psoDesc.pRootSignature = mRootSignature.Get();
    psoDesc.NumRenderTargets = 1;
    psoDesc.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM_SRGB;
    psoDesc.DSVFormat = DEPTH_FORMAT;
    psoDesc.InputLayout.NumElements = std::extent<decltype(layout)>::value;
    psoDesc.InputLayout.pInputElementDescs = layout;
    psoDesc.RasterizerState = CD3DX12_RASTERIZER_DESC(D3D12_DEFAULT);
    // psoDesc.RasterizerState.CullMode = D3D12_CULL_MODE_NONE; // both faces drawn
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
    // let's create default depth testing: DepthEnable = TRUE; DepthFunc = D3D12_COMPARISON_FUNC_LESS;
    psoDesc.DepthStencilState = CD3DX12_DEPTH_STENCIL_DESC(D3D12_DEFAULT);

    mDevice->CreateGraphicsPipelineState(&psoDesc, IID_PPV_ARGS(&mPso));
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
                                         D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&mPistolConstantBuffers[i]));

        mDevice->CreateCommittedResource(&uploadHeapProperties, D3D12_HEAP_FLAG_NONE, &constantBufferDesc,
                                         D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&mMonsterConstantBuffers[i]));

        mDevice->CreateCommittedResource(&uploadHeapProperties, D3D12_HEAP_FLAG_NONE, &constantBufferDesc,
                                         D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&mLandScapeConstantBuffers[i]));

        // we set identity matrix as mvp for models
        void* p;
        mPistolConstantBuffers[i]->Map(0, nullptr, &p);
        memcpy(p, &mConstantBufferData, sizeof(mConstantBufferData));
        mPistolConstantBuffers[i]->Unmap(0, nullptr);

        mMonsterConstantBuffers[i]->Map(0, nullptr, &p);
        memcpy(p, &mConstantBufferData, sizeof(mConstantBufferData));
        mMonsterConstantBuffers[i]->Unmap(0, nullptr);

        mLandScapeConstantBuffers[i]->Map(0, nullptr, &p);
        memcpy(p, &mConstantBufferData, sizeof(mConstantBufferData));
        mLandScapeConstantBuffers[i]->Unmap(0, nullptr);
    }
}

void DirectXRenderer::UpdateConstantBuffer() {
    static auto rotation = XMMatrixRotationQuaternion(_md5RotAdjustment);
    // pistol mvp matrix
    {
        // some offset from camera to our hand&pistol
        static constexpr float offsetFromCamera = 7.0f;
        const auto translation = DirectX::XMMatrixTranslation(0, 0, offsetFromCamera);
        DirectX::XMMATRIX model = DirectX::XMMatrixMultiply(rotation, translation);

        const auto& viewProj = mCamera->viewProjMat();
        // we ignore view matrix because our hand&pistol must follow camera rotation
        DirectX::XMMATRIX modelView = model;
        mConstantBufferData.mvp = DirectX::XMMatrixMultiply(modelView, viewProj.proj);

        void* data;
        mPistolConstantBuffers[m_currentFrame]->Map(0, nullptr, &data);
        memcpy(data, &mConstantBufferData, sizeof(mConstantBufferData));
        mPistolConstantBuffers[m_currentFrame]->Unmap(0, nullptr);
    }

    // monster mvp matrix
    {
        auto rotation = XMMatrixRotationQuaternion(mQuatMonsterRot);
        const auto translation = DirectX::XMMatrixTranslation(mMonsterBasePos.x, mMonsterBasePos.y, mMonsterBasePos.z);
        DirectX::XMMATRIX model = DirectX::XMMatrixMultiply(rotation, translation);

        const auto& viewProj = mCamera->viewProjMat();
        DirectX::XMMATRIX modelView = DirectX::XMMatrixMultiply(model, viewProj.view);
        mConstantBufferData.mvp = DirectX::XMMatrixMultiply(modelView, viewProj.proj);

        void* data;
        mMonsterConstantBuffers[m_currentFrame]->Map(0, nullptr, &data);
        memcpy(data, &mConstantBufferData, sizeof(mConstantBufferData));
        mMonsterConstantBuffers[m_currentFrame]->Unmap(0, nullptr);
    }

    // landscape mvp matrix
    {
        const auto& viewProj = mCamera->viewProjMat();
        // landscape doesn't have model matrix since it's static object
        mConstantBufferData.mvp = DirectX::XMMatrixMultiply(viewProj.view, viewProj.proj);

        void* data;
        mLandScapeConstantBuffers[m_currentFrame]->Map(0, nullptr, &data);
        memcpy(data, &mConstantBufferData, sizeof(mConstantBufferData));
        mLandScapeConstantBuffers[m_currentFrame]->Unmap(0, nullptr);
    }
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

    ComPtr<IDXGIFactory5> factory5;
    ComPtr<IDXGIFactory1> factory1;
    if (SUCCEEDED(CreateDXGIFactory1(IID_PPV_ARGS(&factory1))) && SUCCEEDED(factory1.As(&factory5))) {
        if (FAILED(factory5->CheckFeatureSupport(DXGI_FEATURE_PRESENT_ALLOW_TEARING, &allowTearing, sizeof(allowTearing)))) {
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

    mIsTearingSupport = CheckTearingSupport();

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
    swapChainDesc.Flags = mIsTearingSupport ? DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING : 0;  // set free-sync\g-sync instead v-sync

    // the driver may support directx 12 api but without hardware acceleration
    // D3D_FEATURE_LEVEL_11_0 hardware acceleration is present for sure
    auto renderEnv = CreateDeviceAndSwapChainHelper(nullptr, D3D_FEATURE_LEVEL_11_0, &swapChainDesc);

    mDevice = renderEnv.device;
    mCommandQueue = renderEnv.queue;
    mSwapChain = renderEnv.swapChain;

    mRenderTargetViewDescriptorSize = mDevice->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);

    SetupSwapChain();
}
