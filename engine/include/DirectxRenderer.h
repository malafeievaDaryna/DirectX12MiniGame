#pragma once

#define NOMINMAX

#include <Keyboard.h>
#include <Mouse.h>
#include <d3d12.h>
#include <directxmath.h>
#include <dxgi.h>
#include <wrl.h>  // ComPtr template (kinda smart pointers for COM objects)
#include <memory>
#include <string>
#include <vector>
#include "Camera.h"
#include "MD5Loader.h"
#include "SkyBox.h"
#include "ParticleSystem.h"

class Window;

class DirectXRenderer {
    struct ConstantBuffer {
        DirectX::XMMATRIX mvp;
        DirectX::XMMATRIX model;
        DirectX::XMFLOAT4 lightPos;
        DirectX::XMFLOAT4 lightDir;
        BOOL isMountedToCameraObject{FALSE};
        BOOL isNormalMapInsteadParalaxMapping{TRUE};
        float shootingAnimTimeMS{50.0f};
        float accumulatedTimeMS;
    };

    enum class PISTOL_ANIM { IDLE = 0, RELOAD, FIRE, MAX };
    enum class MONSTER_ANIM { IDLE = 0, RUN, PAIN, MAX };

public:
    DirectXRenderer();
    ~DirectXRenderer();

    void Initialize(const std::string& title, int width, int height);
    bool Run();

private:
    void Shutdown();
    void Render();
    void UpdateConstantBuffer();
    bool CheckTearingSupport();
    void CreateConstantBuffer();
    void CreateMeshBuffers(ID3D12GraphicsCommandList* uploadCommandList);
    void CreateRootSignature();
    void CreatePipelineStateObject();
    void CreateDeviceAndSwapChain();
    void SetupSwapChain();
    void SetupRenderTargets();
    void SetupImGui();

private:
    std::unique_ptr<Window, void (*)(Window*)> mWindow;
    std::unique_ptr<DirectX::Keyboard> mKeyboard{nullptr};
    std::unique_ptr<DirectX::Mouse> mMouse{nullptr};
    std::unique_ptr<Camera> mCamera{nullptr};

    D3D12_VIEWPORT mViewport{};
    D3D12_RECT mRectScissor{};
    Microsoft::WRL::ComPtr<IDXGISwapChain> mSwapChain{};
    Microsoft::WRL::ComPtr<ID3D12Device> mDevice{};
    Microsoft::WRL::ComPtr<ID3D12Resource> mRenderTargets[constants::MAX_FRAMES_IN_FLIGHT]{};
    Microsoft::WRL::ComPtr<ID3D12Resource> mDepthStencilBuffer;
    Microsoft::WRL::ComPtr<ID3D12CommandQueue> mCommandQueue{};
    bool mIsTearingSupport{false};

    HANDLE mFrameFenceEvents[constants::MAX_FRAMES_IN_FLIGHT]{nullptr};
    Microsoft::WRL::ComPtr<ID3D12Fence> mFrameFences[constants::MAX_FRAMES_IN_FLIGHT]{};
    UINT64 mCurrentFenceValue{0u};
    UINT64 mFenceValues[constants::MAX_FRAMES_IN_FLIGHT]{};
    UINT32 m_currentFrame{0u};

    Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> mDSDescriptorHeap;  // the heap for Depth Stencil buffer descriptor
    Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> mRenderTargetDescriptorHeap{};
    UINT64 mRenderTargetViewDescriptorSize{0u};

    Microsoft::WRL::ComPtr<ID3D12RootSignature> mRootSignature{};
    Microsoft::WRL::ComPtr<ID3D12PipelineState> mPso{};

    // TODO move it from here
    // landscape
    Microsoft::WRL::ComPtr<ID3D12Resource> mLandScapeConstantBuffers[constants::MAX_FRAMES_IN_FLIGHT];
    Microsoft::WRL::ComPtr<ID3D12Resource> mUploadBuffer{};
    Microsoft::WRL::ComPtr<ID3D12Resource> mVertexBuffer{};
    D3D12_VERTEX_BUFFER_VIEW mVertexBufferView;
    Microsoft::WRL::ComPtr<ID3D12Resource> mIndexBuffer{};
    D3D12_INDEX_BUFFER_VIEW mIndexBufferView{};
    utils::Texture2DResource mLandscapeTexture;

    Microsoft::WRL::ComPtr<ID3D12CommandAllocator> mCommandAllocators[constants::MAX_FRAMES_IN_FLIGHT]{};
    Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList> mCommandLists[constants::MAX_FRAMES_IN_FLIGHT]{};

    ConstantBuffer mConstantBufferData{};  // temporal projection of gpu memory on cpu accessible memory
    Microsoft::WRL::ComPtr<ID3D12Resource> mPistolConstantBuffers[constants::MAX_FRAMES_IN_FLIGHT];
    std::unique_ptr<MD5Loader> md5PistolModel{};
    Microsoft::WRL::ComPtr<ID3D12Resource> mMonsterConstantBuffers[constants::MAX_FRAMES_IN_FLIGHT];
    std::unique_ptr<MD5Loader> md5MonsterModel{};
    std::unique_ptr<SkyBox> mSkyBox{};

    Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> mImGuiDescriptorHeap;

    PISTOL_ANIM mCurPistolAnim{PISTOL_ANIM::IDLE};
    MONSTER_ANIM mCurMonsterAnim{MONSTER_ANIM::RUN};
    std::unordered_map<PISTOL_ANIM, std::function<void()>> mPistolAnimationsActions;
    std::unordered_map<MONSTER_ANIM, std::function<void()>> mMonsterAnimationsActions;
    DirectX::XMFLOAT3 mMonsterBasePos{.0f, .0f, .0f};
    DirectX::XMVECTOR mQuatMonsterRot = DirectX::XMVectorSet(0.0f, 0.0f, 0.0f, 1.0f);

    std::unique_ptr<ParticleSystem> mGrassParticles{};

    float mFlashLightYOffset{0.0f};

    float mFrameTimeMS{0.0f};
};
