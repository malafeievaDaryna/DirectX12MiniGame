#include "Camera.h"

namespace {
const DirectX::XMVECTOR _forwardDir = DirectX::XMVectorSet(0.0f, 0.0f, 1.0f, 0.0f);
const DirectX::XMVECTOR _leftDir = DirectX::XMVectorSet(-1.0f, 0.0f, 0.0f, 0.0f);
const DirectX::XMVECTOR _upDir = DirectX::XMVectorSet(0.0f, 1.0f, 0.0f, 0.0f);
};  // namespace

using namespace DirectX;

Camera::Camera(const Perstective& perstective, const XMFLOAT4& eye,
               const XMFLOAT4& target)
    : mTarget(XMLoadFloat4(&target)) {
    auto eyeVec = XMLoadFloat4(&eye);
    auto targetVec = XMLoadFloat4(&target);
    mFromTargetToEye = XMVectorSubtract(eyeVec, targetVec);
    mViewProj.view = XMMatrixLookAtLH(eyeVec, targetVec, _upDir);
    mCameraRotationQuat = XMQuaternionRotationNormal(_upDir, XMConvertToRadians(0.0f)); // XMQuaternionRotationAxis is used for non-normalized vector
    resetPerspective(perstective);
}

void Camera::resetPerspective(const Perstective& perstective) {
    mViewProj.proj = XMMatrixPerspectiveFovLH(XMConvertToRadians(perstective.fovy), perstective.aspect,
                                                       perstective._near, perstective._far);
}

void Camera::update(EDirection dir) {
    static const XMVECTOR rotRight =
        XMQuaternionRotationNormal(_upDir, XMConvertToRadians(-1.0f * ANGLE_GAIN));
    static const XMVECTOR rotLeft =
        XMQuaternionRotationNormal(_upDir, XMConvertToRadians(ANGLE_GAIN));

    if (dir == EDirection::Left || dir == EDirection::Right) {
        mCameraRotationQuat = XMQuaternionMultiply(mCameraRotationQuat, ((dir == EDirection::Left) ? rotLeft : rotRight));
    }
    XMFLOAT4 cameraQuatToVector;
    XMStoreFloat4(&cameraQuatToVector, mCameraRotationQuat);
    XMVECTOR cameraRotationQuatConjugate =
        DirectX::XMVectorSet(-cameraQuatToVector.x, -cameraQuatToVector.y, -cameraQuatToVector.z, cameraQuatToVector.w);

    XMVECTOR rotDir = XMQuaternionMultiply(XMQuaternionMultiply(mCameraRotationQuat, _forwardDir),
                                                         cameraRotationQuatConjugate);
    mTarget = XMVectorAdd(mTarget, GAIN_MOVEMENT * ((dir == EDirection::Back) ? (-1.0f * rotDir) : rotDir));

    // fromTargetToEye is multiplied by inverted quat
    XMVECTOR fromTargetToEye =
        XMQuaternionMultiply(XMQuaternionMultiply(cameraRotationQuatConjugate, mFromTargetToEye), mCameraRotationQuat);
    mViewProj.view = DirectX::XMMatrixLookAtLH(XMVectorAdd(mTarget, fromTargetToEye), mTarget, _upDir);

    return;
}
