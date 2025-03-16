#pragma once

#include <d3d12.h>
#include <directxmath.h>
#include <wrl.h>
#include <functional>
#include <memory>
#include <string>
#include <vector>
#include "Utils.h"

using namespace DirectX;
class MD5Loader {
    struct Vertex {
        Vertex() {
        }
        Vertex(float x, float y, float z, float u, float v, float nx, float ny, float nz, float tx, float ty, float tz)
            : pos(x, y, z), texCoord(u, v), normal(nx, ny, nz), tangent(tx, ty, tz) {
        }

        XMFLOAT3 pos;
        XMFLOAT2 texCoord;
        XMFLOAT3 normal;
        XMFLOAT3 tangent;
        XMFLOAT3 biTangent;

        // Will not be sent to shader
        int StartWeight;
        int WeightCount;
    };

    struct Joint {
        std::string name;
        int parentID;

        XMFLOAT3 pos;
        XMFLOAT4 orientation;
    };

    struct BoundingBox {
        XMFLOAT3 min;
        XMFLOAT3 max;
    };

    struct FrameData {
        int frameID;
        std::vector<float> frameData;
    };

    struct AnimJointInfo {
        std::string name;
        int parentID;

        int flags;
        int startIndex;
    };

    struct ModelAnimation {
        int numFrames;
        int numJoints;
        int frameRate;
        int numAnimatedComponents;

        float frameTime;
        float totalAnimTime;
        float currAnimTime;

        std::vector<AnimJointInfo> jointInfo;
        std::vector<BoundingBox> frameBounds;
        std::vector<Joint> baseFrameJoints;
        std::vector<FrameData> frameData;
        std::vector<std::vector<Joint>> frameSkeleton;
    };

    struct Weight {
        int jointID;
        float bias;
        XMFLOAT3 pos;
        XMFLOAT3 normal;
        XMFLOAT3 tangent;
        XMFLOAT3 bitangent;
    };

    struct ModelSubset {
        utils::Texture2DResource texture;
        int numTriangles;

        std::vector<Vertex> vertices;
        std::vector<XMFLOAT3> jointSpaceNormals;
        std::vector<uint32_t> indices;
        std::vector<Weight> weights;

        Microsoft::WRL::ComPtr<ID3D12Resource> verticesBuffer{};
        Microsoft::WRL::ComPtr<ID3D12Resource> indicesBuffer{};
        D3D12_VERTEX_BUFFER_VIEW verticesBufferView;
        D3D12_INDEX_BUFFER_VIEW indicesBufferView{};
    };

    struct Model3D {
        int numSubsets;
        int numJoints;

        std::vector<Joint> joints;
        std::vector<ModelSubset> subsets;
        std::vector<ModelAnimation> animations;
    };

public:
    MD5Loader(ID3D12Device* device, ID3D12GraphicsCommandList* uploadCommandList, const std::string& md5ModelFileName,
              const std::vector<std::string>& md5AnimFileNames);
    void UpdateMD5Model(float deltaTimeMS, int animation = 0u, const std::function<void()>& callBackAnimFinished = nullptr);
    void Draw(ID3D12GraphicsCommandList* commandList);
    const DirectX::XMFLOAT3& GetPosDiffFirstLastFrames() const {
        return mPosDiffFirstLastFrames;
    }
    const float GetRadius() const {
        return mRadius;
    }

private:
    bool LoadMD5Model(ID3D12Device* device, ID3D12GraphicsCommandList* uploadCommandList, const std::string& filename);
    bool LoadMD5Anim(const std::vector<std::string>& filenames);
    void updateAnimationChunk(std::size_t subsetId, std::size_t indexFrom, std::size_t indexTo);
    void calculateInterpolatedSkeleton(std::size_t animationID, std::size_t frame0, std::size_t frame1, float interpolation,
                                       std::size_t indexFrom, std::size_t indexTo);

private:
    Model3D mMD5Model;
    // base intermediate animation as interpolation between neighbor frames animations
    // we keep it in memory to avoid allocations for each frame update
    std::vector<Joint> mInterpolatedSkeleton;
    int mLastAnimationID{0};
    DirectX::XMFLOAT3 mPosDiffFirstLastFrames{.0f, .0f, .0f};
    float mRadius{0.0f};
};
