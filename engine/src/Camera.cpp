#include "Camera.h"

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

void Camera::update(EDirection dir) {
    static const XMVECTOR rotRight = XMQuaternionRotationNormal(_upDir, XMConvertToRadians(-1.0f * ANGLE_GAIN));
    static const XMVECTOR rotLeft = XMQuaternionRotationNormal(_upDir, XMConvertToRadians(ANGLE_GAIN));
    XMVECTOR eye;
    if (dir == EDirection::Left || dir == EDirection::Right) {
        XMVECTOR shiftRight = GAIN_MOVEMENT * XMVector3Normalize(XMVector3Cross(_upDir, mFromEyeToTarget));
        mEye = XMVectorAdd(mEye, (dir == EDirection::Right) ? shiftRight : -1.0f * shiftRight);
        eye = XMVectorAdd(mEye, mFromEyeToTarget);
    } else if (dir == EDirection::Turn_Left || dir == EDirection::Turn_Right) {
        XMVECTOR rot = (dir == EDirection::Turn_Left) ? rotLeft : rotRight;
        mFromEyeToTarget = XMQuaternionMultiply(XMQuaternionMultiply(rot, mFromEyeToTarget), XMQuaternionConjugate(rot));
        eye = XMVectorAdd(mEye, GAIN_MOVEMENT * ((dir == EDirection::Back) ? (-1.0f * mFromEyeToTarget) : mFromEyeToTarget));
    } else {
        mEye = XMVectorAdd(mEye, GAIN_MOVEMENT * ((dir == EDirection::Back) ? (-1.0f * mFromEyeToTarget) : mFromEyeToTarget));
        eye = XMVectorAdd(mEye, mFromEyeToTarget);
    }

    mViewProj.view = DirectX::XMMatrixLookAtLH(mEye, eye, _upDir);

    return;
}
