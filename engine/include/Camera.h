#pragma once

#include <directxmath.h>

class Camera {
public:
    constexpr static float ANGLE_GAIN = 0.25f;
    constexpr static float GAIN_MOVEMENT = 0.25f;

    enum class EDirection { Forward = 0, Left, Right, Back };

    struct Perstective {
        float fovy = 65.0f;
        float aspect = 1.0f;  // width / height
        float _near = 0.01f;
        float _far= 1000.0f;
    };

    struct ViewProj {
        DirectX::XMMATRIX view{};
        DirectX::XMMATRIX proj{};
    };

    Camera(const Perstective& perstective, const DirectX::XMFLOAT4& eye,
           const DirectX::XMFLOAT4& target = DirectX::XMFLOAT4(0.0f, 0.0f, 0.0f, 1.0f));
    void resetPerspective(const Perstective& perstective);

    const ViewProj& viewProjMat() {
        return mViewProj;
    }

    DirectX::XMFLOAT4 cameraPosition() {
        DirectX::XMFLOAT4 pos;
        DirectX::XMStoreFloat4(&pos, mEye);
        return pos;
    }

    void update(EDirection dir);

private:
    DirectX::XMVECTOR mEye;
    DirectX::XMVECTOR mFromEyeToTarget;
    ViewProj mViewProj{};
};