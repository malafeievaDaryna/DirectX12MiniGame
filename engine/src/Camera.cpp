#include "Camera.h"
#include "Utils.h"

namespace {
const DirectX::XMVECTOR _forwardDir = DirectX::XMVectorSet(0.0f, 0.0f, 1.0f, 0.0f);
const DirectX::XMVECTOR _leftDir = DirectX::XMVectorSet(-1.0f, 0.0f, 0.0f, 0.0f);
const DirectX::XMVECTOR _upDir = DirectX::XMVectorSet(0.0f, 1.0f, 0.0f, 0.0f);
};  // namespace

using namespace DirectX;

Camera::Camera(const Perstective& perstective, const XMFLOAT4& eye, const XMFLOAT4& target) {
    mEye = XMLoadFloat4(&eye);
    auto targetVec = XMLoadFloat4(&target);
    mFromEyeToTarget = XMVector3Normalize(XMVectorSubtract(targetVec, mEye));
    mViewProj.view = XMMatrixLookAtLH(mEye, XMVectorAdd(mEye, mFromEyeToTarget), _upDir);
    resetPerspective(perstective);
}

void Camera::resetPerspective(const Perstective& perstective) {
    mViewProj.proj =
        XMMatrixPerspectiveFovLH(XMConvertToRadians(perstective.fovy), perstective.aspect, perstective._near, perstective._far);
}

void Camera::update(float deltaTimeMS, EDirection dir) {
    // camera is complied with 144 FPS
    float frameTimeFactor = deltaTimeMS >= constants::_144_FPS_TO_MS ? 1.0f : (deltaTimeMS / constants::_144_FPS_TO_MS);
    const float angleGain = frameTimeFactor * ANGLE_GAIN;

    const XMVECTOR rotRight = XMQuaternionRotationNormal(_upDir, XMConvertToRadians(-1.0f * angleGain));
    const XMVECTOR rotLeft = XMQuaternionRotationNormal(_upDir, XMConvertToRadians(angleGain));
    XMVECTOR eye;
    const float gainMovement = frameTimeFactor * GAIN_MOVEMENT;
    if (dir == EDirection::Left || dir == EDirection::Right) {
        XMVECTOR shiftRight = gainMovement * XMVector3Normalize(XMVector3Cross(_upDir, mFromEyeToTarget));
        mEye = XMVectorAdd(mEye, (dir == EDirection::Right) ? shiftRight : -1.0f * shiftRight);
        eye = XMVectorAdd(mEye, mFromEyeToTarget);
    } else if (dir == EDirection::Turn_Left || dir == EDirection::Turn_Right) {
        XMVECTOR rot = (dir == EDirection::Turn_Left) ? rotLeft : rotRight;
        mFromEyeToTarget = XMQuaternionMultiply(XMQuaternionMultiply(rot, mFromEyeToTarget), XMQuaternionConjugate(rot));
        eye = XMVectorAdd(mEye, gainMovement * ((dir == EDirection::Back) ? (-1.0f * mFromEyeToTarget) : mFromEyeToTarget));
    } else {
        mEye = XMVectorAdd(mEye, gainMovement * ((dir == EDirection::Back) ? (-1.0f * mFromEyeToTarget) : mFromEyeToTarget));
        eye = XMVectorAdd(mEye, mFromEyeToTarget);
    }

    mViewProj.view = DirectX::XMMatrixLookAtLH(mEye, eye, _upDir);

    return;
}
