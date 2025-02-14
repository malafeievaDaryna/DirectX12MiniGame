#pragma once

#include <directxmath.h>

class Camera {
public:
    constexpr static float ANGLE_GAIN = 3.5f;
    constexpr static float GAIN_MOVEMENT = 2.5f;

    enum class EDirection { Forward = 0, Left, Right, Back, Turn_Left, Turn_Right };

    struct Perstective {
        float fovy = 65.0f;
        float aspect = 1.0f;  // width / height
        float _near = 0.01f;
        float _far = 1000.0f;
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

    const DirectX::XMFLOAT4& cameraPosition() {
        static DirectX::XMFLOAT4 pos;
        DirectX::XMStoreFloat4(&pos, mEye);
        return pos;
    }

    DirectX::XMFLOAT4 targetPosition() {
        DirectX::XMFLOAT4 pos;
        DirectX::XMStoreFloat4(&pos, mFromEyeToTarget);
        return pos;
    }

    void update(float deltaTimeMS, EDirection dir);

private:
    DirectX::XMVECTOR mEye;
    DirectX::XMVECTOR mFromEyeToTarget;
    ViewProj mViewProj{};
};