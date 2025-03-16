#include "MD5Loader.h"
#include <array>
#include <cassert>
#include <fstream>
#include <future>
#include "d3dx12.h"

MD5Loader::MD5Loader(ID3D12Device* device, ID3D12GraphicsCommandList* uploadCommandList, const std::string& md5ModelFileName,
                     const std::vector<std::string>& md5AnimFileNames) {
    assert(device && uploadCommandList);
    LoadMD5Model(device, uploadCommandList, md5ModelFileName);
    LoadMD5Anim(md5AnimFileNames);
}

bool MD5Loader::LoadMD5Anim(const std::vector<std::string>& filenames) {
    bool result = true;
    for (const auto& filename : filenames) {
        ModelAnimation tempAnim;  // Temp animation to later store in our model's animation array

        std::ifstream fileIn(filename.c_str());  // Open file

        std::string checkString;  // Stores the next string from our file

        if (fileIn) {
            while (fileIn)  // Loop until the end of the file is reached
            {
                fileIn >> checkString;  // Get next string from file

                if (checkString == "MD5Version")  // Get MD5 version (this function supports version 10)
                {
                    fileIn >> checkString;
                } else if (checkString == "commandline") {
                    std::getline(fileIn, checkString);  // Ignore the rest of this line
                } else if (checkString == "numFrames") {
                    fileIn >> tempAnim.numFrames;  // Store number of frames in this animation
                } else if (checkString == "numJoints") {
                    fileIn >> tempAnim.numJoints;  // Store number of joints (must match .md5mesh)
                } else if (checkString == "frameRate") {
                    fileIn >> tempAnim.frameRate;  // Store animation's frame rate (frames per second)
                } else if (checkString == "numAnimatedComponents") {
                    fileIn >> tempAnim.numAnimatedComponents;  // Number of components in each frame section
                } else if (checkString == "hierarchy") {
                    fileIn >> checkString;  // Skip opening bracket "{"

                    for (int i = 0; i < tempAnim.numJoints; i++)  // Load in each joint
                    {
                        AnimJointInfo tempJoint;

                        fileIn >> tempJoint.name;  // Get joints name
                        // Sometimes the names might contain spaces. If that is the case, we need to continue
                        // to read the name until we get to the closing " (quotation marks)
                        if (tempJoint.name[tempJoint.name.size() - 1] != '"') {
                            char checkChar;
                            bool jointNameFound = false;
                            while (!jointNameFound) {
                                checkChar = fileIn.get();

                                if (checkChar == '"')
                                    jointNameFound = true;

                                tempJoint.name += checkChar;
                            }
                        }

                        // Remove the quotation marks from joints name
                        tempJoint.name.erase(0, 1);
                        tempJoint.name.erase(tempJoint.name.size() - 1, 1);

                        fileIn >> tempJoint.parentID;    // Get joints parent ID
                        fileIn >> tempJoint.flags;       // Get flags
                        fileIn >> tempJoint.startIndex;  // Get joints start index

                        // Make sure the joint exists in the model, and the parent ID's match up
                        // because the bind pose (md5mesh) joint hierarchy and the animations (md5anim)
                        // joint hierarchy must match up
                        bool jointMatchFound = false;
                        tempAnim.jointInfo.reserve(mMD5Model.numJoints);
                        for (int k = 0; k < mMD5Model.numJoints; k++) {
                            if (mMD5Model.joints[k].name == tempJoint.name) {
                                if (mMD5Model.joints[k].parentID == tempJoint.parentID) {
                                    jointMatchFound = true;
                                    tempAnim.jointInfo.push_back(tempJoint);
                                }
                            }
                        }
                        if (!jointMatchFound)  // If the skeleton system does not match up, return false
                            return false;      // You might want to add an error message here

                        std::getline(fileIn, checkString);  // Skip rest of this line
                    }
                } else if (checkString == "bounds")  // Load in the AABB for each animation
                {
                    fileIn >> checkString;  // Skip opening bracket "{"

                    tempAnim.frameBounds.reserve(tempAnim.numFrames);
                    for (int i = 0; i < tempAnim.numFrames; i++) {
                        BoundingBox tempBB;

                        fileIn >> checkString;  // Skip "("
                        fileIn >> tempBB.min.x >> tempBB.min.z >> tempBB.min.y;
                        fileIn >> checkString >> checkString;  // Skip ") ("
                        fileIn >> tempBB.max.x >> tempBB.max.z >> tempBB.max.y;
                        fileIn >> checkString;  // Skip ")"

                        if (std::numeric_limits<float>::epsilon() >= mRadius) {
                            DirectX::XMVECTOR min = DirectX::XMVectorSet(tempBB.min.x, tempBB.min.y, tempBB.min.z, 0.0f);
                            DirectX::XMVECTOR max = DirectX::XMVectorSet(tempBB.max.x, tempBB.max.y, tempBB.max.z, 0.0f);
                            DirectX::XMVECTOR lengthVec = XMVector3Length(XMVectorSubtract(max, min));
                            XMFLOAT4 length;
                            XMStoreFloat4(&length, lengthVec);
                            mRadius = length.x / 2.0f;
                        }

                        tempAnim.frameBounds.push_back(tempBB);
                    }
                } else if (checkString == "baseframe")  // This is the default position for the animation
                {                                       // All frames will build their skeletons off this
                    fileIn >> checkString;              // Skip opening bracket "{"

                    tempAnim.baseFrameJoints.reserve(tempAnim.numJoints);
                    for (int i = 0; i < tempAnim.numJoints; i++) {
                        Joint tempBFJ;

                        fileIn >> checkString;  // Skip "("
                        fileIn >> tempBFJ.pos.x >> tempBFJ.pos.z >> tempBFJ.pos.y;
                        fileIn >> checkString >> checkString;  // Skip ") ("
                        fileIn >> tempBFJ.orientation.x >> tempBFJ.orientation.z >> tempBFJ.orientation.y;
                        fileIn >> checkString;  // Skip ")"

                        tempAnim.baseFrameJoints.push_back(tempBFJ);
                    }
                } else if (checkString ==
                           "frame")  // Load in each frames skeleton (the parts of each joint that changed from the base frame)
                {
                    FrameData tempFrame;

                    fileIn >> tempFrame.frameID;  // Get the frame ID

                    fileIn >> checkString;  // Skip opening bracket "{"

                    tempFrame.frameData.reserve(tempAnim.numAnimatedComponents);
                    for (int i = 0; i < tempAnim.numAnimatedComponents; i++) {
                        float tempData;
                        fileIn >> tempData;  // Get the data

                        tempFrame.frameData.push_back(tempData);
                    }

                    tempAnim.frameData.push_back(tempFrame);

                    ///*** build the frame skeleton ***///
                    std::vector<Joint> tempSkeleton;

                    tempSkeleton.reserve(tempAnim.jointInfo.size());
                    for (int i = 0; i < tempAnim.jointInfo.size(); i++) {
                        int k = 0;  // Keep track of position in frameData array

                        // Start the frames joint with the base frame's joint
                        Joint tempFrameJoint = tempAnim.baseFrameJoints[i];

                        tempFrameJoint.parentID = tempAnim.jointInfo[i].parentID;

                        // Notice how I have been flipping y and z. this is because some modeling programs such as
                        // 3ds max (which is what I use) use a right handed coordinate system. Because of this, we
                        // need to flip the y and z axes. If your having problems loading some models, it's possible
                        // the model was created in a left hand coordinate system. in that case, just reflip all the
                        // y and z axes in our md5 mesh and anim loader.
                        if (tempAnim.jointInfo[i].flags & 1)  // pos.x	( 000001 )
                            tempFrameJoint.pos.x = tempFrame.frameData[tempAnim.jointInfo[i].startIndex + k++];

                        if (tempAnim.jointInfo[i].flags & 2)  // pos.y	( 000010 )
                            tempFrameJoint.pos.z = tempFrame.frameData[tempAnim.jointInfo[i].startIndex + k++];

                        if (tempAnim.jointInfo[i].flags & 4)  // pos.z	( 000100 )
                            tempFrameJoint.pos.y = tempFrame.frameData[tempAnim.jointInfo[i].startIndex + k++];

                        if (tempAnim.jointInfo[i].flags & 8)  // orientation.x	( 001000 )
                            tempFrameJoint.orientation.x = tempFrame.frameData[tempAnim.jointInfo[i].startIndex + k++];

                        if (tempAnim.jointInfo[i].flags & 16)  // orientation.y	( 010000 )
                            tempFrameJoint.orientation.z = tempFrame.frameData[tempAnim.jointInfo[i].startIndex + k++];

                        if (tempAnim.jointInfo[i].flags & 32)  // orientation.z	( 100000 )
                            tempFrameJoint.orientation.y = tempFrame.frameData[tempAnim.jointInfo[i].startIndex + k++];

                        // Compute the quaternions w
                        float t = 1.0f - (tempFrameJoint.orientation.x * tempFrameJoint.orientation.x) -
                                  (tempFrameJoint.orientation.y * tempFrameJoint.orientation.y) -
                                  (tempFrameJoint.orientation.z * tempFrameJoint.orientation.z);
                        if (t < 0.0f) {
                            tempFrameJoint.orientation.w = 0.0f;
                        } else {
                            tempFrameJoint.orientation.w = -sqrtf(t);
                        }

                        // Now, if the upper arm of your skeleton moves, you need to also move the lower part of your arm, and
                        // then the hands, and then finally the fingers (possibly weapon or tool too) This is where joint
                        // hierarchy comes in. We start at the top of the hierarchy, and move down to each joints child, rotating
                        // and translating them based on their parents rotation and translation. We can assume that by the time we
                        // get to the child, the parent has already been rotated and transformed based of it's parent. We can
                        // assume this because the child should never come before the parent in the files we loaded in.
                        if (tempFrameJoint.parentID >= 0) {
                            Joint parentJoint = tempSkeleton[tempFrameJoint.parentID];

                            // Turn the XMFLOAT3 and 4's into vectors for easier computation
                            XMVECTOR parentJointOrientation = XMVectorSet(parentJoint.orientation.x, parentJoint.orientation.y,
                                                                          parentJoint.orientation.z, parentJoint.orientation.w);
                            XMVECTOR tempJointPos =
                                XMVectorSet(tempFrameJoint.pos.x, tempFrameJoint.pos.y, tempFrameJoint.pos.z, 0.0f);
                            XMVECTOR parentOrientationConjugate =
                                XMVectorSet(-parentJoint.orientation.x, -parentJoint.orientation.y, -parentJoint.orientation.z,
                                            parentJoint.orientation.w);

                            // Calculate current joints position relative to its parents position
                            XMFLOAT3 rotatedPos;
                            XMStoreFloat3(&rotatedPos,
                                          XMQuaternionMultiply(XMQuaternionMultiply(parentJointOrientation, tempJointPos),
                                                               parentOrientationConjugate));

                            // Translate the joint to model space by adding the parent joint's pos to it
                            tempFrameJoint.pos.x = rotatedPos.x + parentJoint.pos.x;
                            tempFrameJoint.pos.y = rotatedPos.y + parentJoint.pos.y;
                            tempFrameJoint.pos.z = rotatedPos.z + parentJoint.pos.z;

                            // Currently the joint is oriented in its parent joints space, we now need to orient it in
                            // model space by multiplying the two orientations together (parentOrientation * childOrientation) <-
                            // In that order
                            XMVECTOR tempJointOrient = XMVectorSet(tempFrameJoint.orientation.x, tempFrameJoint.orientation.y,
                                                                   tempFrameJoint.orientation.z, tempFrameJoint.orientation.w);
                            tempJointOrient = XMQuaternionMultiply(parentJointOrientation, tempJointOrient);

                            // Normalize the orienation quaternion
                            tempJointOrient = XMQuaternionNormalize(tempJointOrient);

                            XMStoreFloat4(&tempFrameJoint.orientation, tempJointOrient);
                        }

                        // Store the joint into our temporary frame skeleton
                        tempSkeleton.push_back(tempFrameJoint);
                    }

                    // Push back our newly created frame skeleton into the animation's frameSkeleton array
                    tempAnim.frameSkeleton.push_back(tempSkeleton);

                    fileIn >> checkString;  // Skip closing bracket "}"
                }
            }

            // Calculate and store some usefull animation data
            tempAnim.frameTime = 1.0f / tempAnim.frameRate;                    // Set the time per frame
            tempAnim.totalAnimTime = tempAnim.numFrames * tempAnim.frameTime;  // Set the total time the animation takes
            tempAnim.currAnimTime = 0.0f;                                      // Set the current time to zero

            mMD5Model.animations.push_back(tempAnim);  // Push back the animation into our model object
        } else                                         // If the file was not loaded
        {
            utils::log_err("Couldn't open file");
            result = false;
            break;
        }
    }
    return result;
}

void MD5Loader::UpdateMD5Model(float deltaTimeMS, int animation, const std::function<void()>& callBackAnimFinished) {
    if (mMD5Model.animations.size() <= animation) {
        utils::log_err("wrong parameters");
        return;
    }

    if (mLastAnimationID != animation) {
        mLastAnimationID = animation;
        mMD5Model.animations[animation].currAnimTime = 0;
    }

    mMD5Model.animations[animation].currAnimTime += deltaTimeMS / 1000.0f;  // Update the current animation time

    /** Note: do nothing since we have another condition for this case
     * if (mMD5Model.animations[animation].currAnimTime >= mMD5Model.animations[animation].totalAnimTime) {
     *
     * }
     */

    static std::function<void(DirectX::XMFLOAT3&, const BoundingBox&, const BoundingBox&, float)> culculateCurrentPos =
        [](DirectX::XMFLOAT3& result, const BoundingBox& firstFrame, const BoundingBox& lastFrame, float interpolation) {
            result.x = lastFrame.max.x - interpolation * firstFrame.max.x;
            result.z = lastFrame.max.z - interpolation * firstFrame.max.z;
            result.y = lastFrame.max.y - interpolation * firstFrame.max.y;
        };

    // Which frame are we on
    float currentFrame = mMD5Model.animations[animation].currAnimTime * mMD5Model.animations[animation].frameRate;
    int frame0 = floorf(currentFrame);
    int frame1 = frame0 + 1;

    float interpolation = 1.0f;

    // Make sure we don't go over the number of frames
    if (frame0 >= mMD5Model.animations[animation].numFrames - 1) {
        mMD5Model.animations[animation].currAnimTime = 0;
        currentFrame = 0;
        frame0 = 0;
        frame1 = 1;

        culculateCurrentPos(mPosDiffFirstLastFrames, mMD5Model.animations[animation].frameBounds[0],
                            mMD5Model.animations[animation].frameBounds[mMD5Model.animations[animation].numFrames - 1], 1.0f);

        if (callBackAnimFinished)
            callBackAnimFinished();
    } else {
        interpolation =
            currentFrame - frame0;  // Get the remainder (in time) between frame0 and frame1 to use as interpolation factor

        culculateCurrentPos(mPosDiffFirstLastFrames, mMD5Model.animations[animation].frameBounds[frame0],
                            mMD5Model.animations[animation].frameBounds[frame1], interpolation);
    }

    // Create a frame skeleton to store the interpolated skeletons in
    if (mInterpolatedSkeleton.size() < mMD5Model.animations[animation].numJoints) {
        mInterpolatedSkeleton.resize(mMD5Model.animations[animation].numJoints);
    }

    std::array<std::future<void>, 4u> workerThreads;
    std::size_t chunkOffset{0u};
    std::size_t indexFrom{0u};
    std::size_t indexTo{0u};
    std::size_t workerThreadIndexPlusOne{0u};

    // Compute the interpolated skeleton in worker_threads
    chunkOffset = mMD5Model.animations[animation].numJoints / workerThreads.size();
    for (std::size_t workerThreadIndex = 0u; workerThreadIndex < workerThreads.size(); ++workerThreadIndex) {
        workerThreadIndexPlusOne = workerThreadIndex + 1U;
        indexFrom = workerThreadIndex * chunkOffset;
        indexTo = workerThreadIndexPlusOne >= workerThreads.size() ? mMD5Model.animations[animation].numJoints
                                                                   : workerThreadIndexPlusOne * chunkOffset;
        workerThreads[workerThreadIndex] = std::async(std::launch::async, &MD5Loader::calculateInterpolatedSkeleton, this,
                                                      animation, frame0, frame1, interpolation, indexFrom, indexTo);
    }
    for (auto& thread : workerThreads) {
        thread.wait();
    }

    for (int k = 0; k < mMD5Model.numSubsets; k++) {
        chunkOffset = mMD5Model.subsets[k].vertices.size() / workerThreads.size();
        for (std::size_t workerThreadIndex = 0u; workerThreadIndex < workerThreads.size(); ++workerThreadIndex) {
            workerThreadIndexPlusOne = workerThreadIndex + 1U;
            indexFrom = workerThreadIndex * chunkOffset;
            indexTo = workerThreadIndexPlusOne >= workerThreads.size() ? mMD5Model.subsets[k].vertices.size()
                                                                       : workerThreadIndexPlusOne * chunkOffset;
            workerThreads[workerThreadIndex] =
                std::async(std::launch::async, &MD5Loader::updateAnimationChunk, this, k, indexFrom, indexTo);
        }

        for (auto& thread : workerThreads) {
            thread.wait();
        }

        // Update the subsets vertex buffer
        const int32_t indicesSize = sizeof(uint32_t) * mMD5Model.subsets[k].numTriangles * 3;
        const int32_t verticesSize = sizeof(Vertex) * mMD5Model.subsets[k].vertices.size();
        void* data;
        mMD5Model.subsets[k].indicesBuffer->Map(0, nullptr, &data);
        memcpy(data, mMD5Model.subsets[k].indices.data(), indicesSize);
        mMD5Model.subsets[k].indicesBuffer->Unmap(0, nullptr);

        mMD5Model.subsets[k].verticesBuffer->Map(0, nullptr, &data);
        memcpy(data, mMD5Model.subsets[k].vertices.data(), verticesSize);
        mMD5Model.subsets[k].verticesBuffer->Unmap(0, nullptr);
    }
}

void MD5Loader::calculateInterpolatedSkeleton(std::size_t animationID, std::size_t frame0, std::size_t frame1,
                                              float interpolation, std::size_t indexFrom, std::size_t indexTo) {
    ModelAnimation& animation = mMD5Model.animations[animationID];
    assert(indexFrom < animation.numJoints && indexTo <= animation.numJoints && indexTo <= mInterpolatedSkeleton.size() &&
           animation.frameSkeleton.size() > frame0 && animation.frameSkeleton.size() > frame1);
    Joint joint0;
    Joint joint1;
    for (std::size_t i = indexFrom; i < indexTo; i++) {
        Joint& tempJoint = mInterpolatedSkeleton[i];
        joint0 = animation.frameSkeleton[frame0][i];  // Get the i'th joint of frame0's skeleton
        joint1 = animation.frameSkeleton[frame1][i];  // Get the i'th joint of frame1's skeleton

        tempJoint.parentID = joint0.parentID;  // Set the tempJoints parent id

        // Turn the two quaternions into XMVECTORs for easy computations
        XMVECTOR joint0Orient =
            XMVectorSet(joint0.orientation.x, joint0.orientation.y, joint0.orientation.z, joint0.orientation.w);
        XMVECTOR joint1Orient =
            XMVectorSet(joint1.orientation.x, joint1.orientation.y, joint1.orientation.z, joint1.orientation.w);

        // Interpolate positions
        tempJoint.pos.x = joint0.pos.x + (interpolation * (joint1.pos.x - joint0.pos.x));
        tempJoint.pos.y = joint0.pos.y + (interpolation * (joint1.pos.y - joint0.pos.y));
        tempJoint.pos.z = joint0.pos.z + (interpolation * (joint1.pos.z - joint0.pos.z));

        // Interpolate orientations using spherical interpolation (Slerp)
        XMStoreFloat4(&tempJoint.orientation, XMQuaternionSlerp(joint0Orient, joint1Orient, interpolation));
    }
}

void MD5Loader::updateAnimationChunk(std::size_t subsetId, std::size_t indexFrom, std::size_t indexTo) {
    ModelSubset& subset = mMD5Model.subsets[subsetId];
    assert(indexFrom < subset.vertices.size() && indexTo <= subset.vertices.size());

    XMFLOAT3 rotatedPoint;
    for (std::size_t i = indexFrom; i < indexTo; ++i) {
        Vertex& tempVert = subset.vertices[i];
        tempVert.pos = XMFLOAT3(0, 0, 0);     // Make sure the vertex's pos is cleared first
        tempVert.normal = XMFLOAT3(0, 0, 0);  // Clear vertices normal
        tempVert.tangent = XMFLOAT3(0, 0, 0);  // Clear vertices normal
        tempVert.biTangent = XMFLOAT3(0, 0, 0);  // Clear vertices normal

        // Sum up the joints and weights information to get vertex's position and normal
        for (std::size_t j = 0; j < tempVert.WeightCount; ++j) {
            const Weight& tempWeight = subset.weights[tempVert.StartWeight + j];
            const Joint& tempJoint = mInterpolatedSkeleton[tempWeight.jointID];

            // Convert joint orientation and weight pos to vectors for easier computation
            XMVECTOR tempJointOrientation =
                XMVectorSet(tempJoint.orientation.x, tempJoint.orientation.y, tempJoint.orientation.z, tempJoint.orientation.w);
            XMVECTOR tempWeightPos = XMVectorSet(tempWeight.pos.x, tempWeight.pos.y, tempWeight.pos.z, 0.0f);

            // We will need to use the conjugate of the joint orientation quaternion
            XMVECTOR tempJointOrientationConjugate = XMQuaternionInverse(tempJointOrientation);

            // Calculate vertex position (in joint space, eg. rotate the point around (0,0,0)) for this weight using the joint
            // orientation quaternion and its conjugate We can rotate a point using a quaternion with the equation
            // "rotatedPoint = quaternion * point * quaternionConjugate"
            XMStoreFloat3(&rotatedPoint, XMQuaternionMultiply(XMQuaternionMultiply(tempJointOrientation, tempWeightPos),
                                                              tempJointOrientationConjugate));

            // Now move the verices position from joint space (0,0,0) to the joints position in world space, taking the
            // weights bias into account
            tempVert.pos.x += (tempJoint.pos.x + rotatedPoint.x) * tempWeight.bias;
            tempVert.pos.y += (tempJoint.pos.y + rotatedPoint.y) * tempWeight.bias;
            tempVert.pos.z += (tempJoint.pos.z + rotatedPoint.z) * tempWeight.bias;

            // Compute the normals for this frames skeleton using the weight normals from before
            // We can comput the normals the same way we compute the vertices position, only we don't have to translate them
            // (just rotate)
            XMVECTOR tempWeightNormal = XMVectorSet(tempWeight.normal.x, tempWeight.normal.y, tempWeight.normal.z, 0.0f);

            // Rotate the normal
            XMStoreFloat3(&rotatedPoint, XMQuaternionMultiply(XMQuaternionMultiply(tempJointOrientation, tempWeightNormal),
                                                              tempJointOrientationConjugate));

            // Add to vertices normal and ake weight bias into account
            tempVert.normal.x -= rotatedPoint.x * tempWeight.bias;
            tempVert.normal.y -= rotatedPoint.y * tempWeight.bias;
            tempVert.normal.z -= rotatedPoint.z * tempWeight.bias;

            XMVECTOR tempWeightTangent = XMVectorSet(tempWeight.tangent.x, tempWeight.tangent.y, tempWeight.tangent.z, 0.0f);
            XMStoreFloat3(&rotatedPoint, XMQuaternionMultiply(XMQuaternionMultiply(tempJointOrientation, tempWeightTangent),
                                                              tempJointOrientationConjugate));

            tempVert.tangent.x -= rotatedPoint.x * tempWeight.bias;
            tempVert.tangent.y -= rotatedPoint.y * tempWeight.bias;
            tempVert.tangent.z -= rotatedPoint.z * tempWeight.bias;

            XMVECTOR tempWeightBitangent =
                XMVectorSet(tempWeight.bitangent.x, tempWeight.bitangent.y, tempWeight.bitangent.z, 0.0f);
            XMStoreFloat3(&rotatedPoint, XMQuaternionMultiply(XMQuaternionMultiply(tempJointOrientation, tempWeightBitangent),
                                                              tempJointOrientationConjugate));

            tempVert.biTangent.x -= rotatedPoint.x * tempWeight.bias;
            tempVert.biTangent.y -= rotatedPoint.y * tempWeight.bias;
            tempVert.biTangent.z -= rotatedPoint.z * tempWeight.bias;
        }

        XMStoreFloat3(&tempVert.normal, XMVector3Normalize(XMLoadFloat3(&tempVert.normal)));
        XMStoreFloat3(&tempVert.tangent, XMVector3Normalize(XMLoadFloat3(&tempVert.tangent)));
        XMStoreFloat3(&tempVert.biTangent, XMVector3Normalize(XMLoadFloat3(&tempVert.biTangent)));
    }
}

bool MD5Loader::LoadMD5Model(ID3D12Device* device, ID3D12GraphicsCommandList* uploadCommandList, const std::string& filename) {
    assert(device && uploadCommandList);
    std::ifstream fileIn(filename.c_str());  // Open file

    std::string checkString;  // Stores the next string from our file

    if (fileIn)  // Check if the file was opened
    {
        while (fileIn)  // Loop until the end of the file is reached
        {
            fileIn >> checkString;  // Get next string from file

            if (checkString == "MD5Version")  // Get MD5 version (this function supports version 10)
            {
            } else if (checkString == "commandline") {
                std::getline(fileIn, checkString);  // Ignore the rest of this line
            } else if (checkString == "numJoints") {
                fileIn >> mMD5Model.numJoints;  // Store number of joints
                mMD5Model.joints.reserve(mMD5Model.numJoints);
            } else if (checkString == "numMeshes") {
                fileIn >> mMD5Model.numSubsets;  // Store number of meshes or subsets which we will call them
                mMD5Model.subsets.reserve(mMD5Model.numSubsets);
            } else if (checkString == "joints") {
                Joint tempJoint;

                fileIn >> checkString;  // Skip the "{"

                for (int i = 0; i < mMD5Model.numJoints; i++) {
                    fileIn >> tempJoint.name;  // Store joints name
                    // Sometimes the names might contain spaces. If that is the case, we need to continue
                    // to read the name until we get to the closing " (quotation marks)
                    if (tempJoint.name[tempJoint.name.size() - 1] != '"') {
                        char checkChar;
                        bool jointNameFound = false;
                        while (!jointNameFound) {
                            checkChar = fileIn.get();

                            if (checkChar == '"')
                                jointNameFound = true;

                            tempJoint.name += checkChar;
                        }
                    }

                    fileIn >> tempJoint.parentID;  // Store Parent joint's ID

                    fileIn >> checkString;  // Skip the "("

                    // Store position of this joint (swap y and z axis if model was made in RH Coord Sys)
                    fileIn >> tempJoint.pos.x >> tempJoint.pos.z >> tempJoint.pos.y;

                    fileIn >> checkString >> checkString;  // Skip the ")" and "("

                    // Store orientation of this joint
                    fileIn >> tempJoint.orientation.x >> tempJoint.orientation.z >> tempJoint.orientation.y;

                    // Remove the quotation marks from joints name
                    tempJoint.name.erase(0, 1);
                    tempJoint.name.erase(tempJoint.name.size() - 1, 1);

                    // Compute the w axis of the quaternion (The MD5 model uses a 3D vector to describe the
                    // direction the bone is facing. However, we need to turn this into a quaternion, and the way
                    // quaternions work, is the xyz values describe the axis of rotation, while the w is a value
                    // between 0 and 1 which describes the angle of rotation)
                    float t = 1.0f - (tempJoint.orientation.x * tempJoint.orientation.x) -
                              (tempJoint.orientation.y * tempJoint.orientation.y) -
                              (tempJoint.orientation.z * tempJoint.orientation.z);
                    if (t < 0.0f) {
                        tempJoint.orientation.w = 0.0f;
                    } else {
                        tempJoint.orientation.w = -sqrtf(t);
                    }

                    std::getline(fileIn, checkString);  // Skip rest of this line

                    mMD5Model.joints.push_back(tempJoint);  // Store the joint into this models joint vector
                }

                fileIn >> checkString;  // Skip the "}"
            } else if (checkString == "mesh") {
                mMD5Model.subsets.emplace_back();
                ModelSubset& subset = mMD5Model.subsets.back();
                int numVerts, numTris, numWeights;

                fileIn >> checkString;  // Skip the "{"

                fileIn >> checkString;
                while (checkString != "}")  // Read until '}'
                {
                    if (checkString == "shader")  // Load the texture
                    {
                        std::string fileNamePath;
                        fileIn >> fileNamePath;  // Get texture's filename

                        // Take spaces into account if filename or material name has a space in it
                        if (fileNamePath[fileNamePath.size() - 1] != '"') {
                            char checkChar;
                            bool fileNameFound = false;
                            while (!fileNameFound) {
                                checkChar = fileIn.get();

                                if (checkChar == '"')
                                    fileNameFound = true;

                                fileNamePath += checkChar;
                            }
                        }

                        // Remove the quotation marks from texture path
                        fileNamePath.erase(0, 1);
                        fileNamePath.erase(fileNamePath.size() - 1, 1);

                        std::string fileNameBumpTexPath = fileNamePath;
                        std::size_t dotIndex = fileNameBumpTexPath.rfind('.');
                        if (dotIndex != std::string::npos) {
                            fileNameBumpTexPath.replace(dotIndex, 1, "_b.");
                        }

                        std::string fileNameSpecularTexPath = fileNamePath;
                        dotIndex = fileNameSpecularTexPath.rfind('.');
                        if (dotIndex != std::string::npos) {
                            fileNameSpecularTexPath.replace(dotIndex, 1, "_s.");
                        }

                        subset.texture = utils::CreateTexture(device, uploadCommandList,
                                                              {fileNamePath, fileNameBumpTexPath, fileNameSpecularTexPath});

                        std::getline(fileIn, checkString);  // Skip rest of this line
                    } else if (checkString == "numverts") {
                        fileIn >> numVerts;  // Store number of vertices

                        std::getline(fileIn, checkString);  // Skip rest of this line

                        subset.vertices.reserve(numVerts);
                        for (int i = 0; i < numVerts; i++) {
                            subset.vertices.emplace_back();
                            Vertex& tempVert = subset.vertices.back();

                            fileIn >> checkString  // Skip "vert # ("
                                >> checkString >> checkString;

                            fileIn >> tempVert.texCoord.x  // Store tex coords
                                >> tempVert.texCoord.y;

                            fileIn >> checkString;  // Skip ")"

                            fileIn >> tempVert.StartWeight;  // Index of first weight this vert will be weighted to

                            fileIn >> tempVert.WeightCount;  // Number of weights for this vertex

                            std::getline(fileIn, checkString);  // Skip rest of this line
                        }
                    } else if (checkString == "numtris") {
                        fileIn >> numTris;
                        subset.numTriangles = numTris;

                        std::getline(fileIn, checkString);  // Skip rest of this line

                        subset.indices.reserve(numTris * 3u);
                        for (int i = 0; i < numTris; i++)  // Loop through each triangle
                        {
                            uint32_t tempIndex;
                            fileIn >> checkString;  // Skip "tri"
                            fileIn >> checkString;  // Skip tri counter

                            for (int k = 0; k < 3; k++)  // Store the 3 indices
                            {
                                fileIn >> tempIndex;
                                subset.indices.push_back(tempIndex);
                            }

                            std::getline(fileIn, checkString);  // Skip rest of this line
                        }
                    } else if (checkString == "numweights") {
                        fileIn >> numWeights;

                        std::getline(fileIn, checkString);  // Skip rest of this line

                        subset.weights.reserve(numWeights);
                        for (int i = 0; i < numWeights; i++) {
                            Weight tempWeight;
                            fileIn >> checkString >> checkString;  // Skip "weight #"

                            fileIn >> tempWeight.jointID;  // Store weight's joint ID

                            fileIn >> tempWeight.bias;  // Store weight's influence over a vertex

                            fileIn >> checkString;  // Skip "("

                            fileIn >> tempWeight.pos.x  // Store weight's pos in joint's local space
                                >> tempWeight.pos.z >> tempWeight.pos.y;

                            std::getline(fileIn, checkString);  // Skip rest of this line

                            subset.weights.push_back(tempWeight);  // Push back tempWeight into subsets Weight array
                        }

                    } else
                        std::getline(fileIn, checkString);  // Skip anything else

                    fileIn >> checkString;  // Skip "}"
                }

                //*** find each vertex's position using the joints and weights ***//
                for (int i = 0; i < subset.vertices.size(); ++i) {
                    Vertex& tempVert = subset.vertices[i];
                    tempVert.pos = XMFLOAT3(0, 0, 0);  // Make sure the vertex's pos is cleared first

                    // Sum up the joints and weights information to get vertex's position
                    for (int j = 0; j < tempVert.WeightCount; ++j) {
                        Weight& tempWeight = subset.weights[tempVert.StartWeight + j];
                        Joint& tempJoint = mMD5Model.joints[tempWeight.jointID];

                        // Convert joint orientation and weight pos to vectors for easier computation
                        // When converting a 3d vector to a quaternion, you should put 0 for "w", and
                        // When converting a quaternion to a 3d vector, you can just ignore the "w"
                        XMVECTOR tempJointOrientation = XMVectorSet(tempJoint.orientation.x, tempJoint.orientation.y,
                                                                    tempJoint.orientation.z, tempJoint.orientation.w);
                        XMVECTOR tempWeightPos = XMVectorSet(tempWeight.pos.x, tempWeight.pos.y, tempWeight.pos.z, 0.0f);

                        // We will need to use the conjugate of the joint orientation quaternion
                        // To get the conjugate of a quaternion, all you have to do is inverse the x, y, and z
                        XMVECTOR tempJointOrientationConjugate = XMVectorSet(-tempJoint.orientation.x, -tempJoint.orientation.y,
                                                                             -tempJoint.orientation.z, tempJoint.orientation.w);

                        // Calculate vertex position (in joint space, eg. rotate the point around (0,0,0)) for this weight using
                        // the joint orientation quaternion and its conjugate We can rotate a point using a quaternion with the
                        // equation "rotatedPoint = quaternion * point * quaternionConjugate"
                        XMFLOAT3 rotatedPoint;
                        XMStoreFloat3(&rotatedPoint,
                                      XMQuaternionMultiply(XMQuaternionMultiply(tempJointOrientation, tempWeightPos),
                                                           tempJointOrientationConjugate));

                        // Now move the verices position from joint space (0,0,0) to the joints position in world space, taking
                        // the weights bias into account The weight bias is used because multiple weights might have an effect on
                        // the vertices final position. Each weight is attached to one joint.
                        tempVert.pos.x += (tempJoint.pos.x + rotatedPoint.x) * tempWeight.bias;
                        tempVert.pos.y += (tempJoint.pos.y + rotatedPoint.y) * tempWeight.bias;
                        tempVert.pos.z += (tempJoint.pos.z + rotatedPoint.z) * tempWeight.bias;

                        // Basically what has happened above, is we have taken the weights position relative to the joints
                        // position we then rotate the weights position (so that the weight is actually being rotated around (0,
                        // 0, 0) in world space) using the quaternion describing the joints rotation. We have stored this rotated
                        // point in rotatedPoint, which we then add to the joints position (because we rotated the weight's
                        // position around (0,0,0) in world space, and now need to translate it so that it appears to have been
                        // rotated around the joints position). Finally we multiply the answer with the weights bias, or how much
                        // control the weight has over the final vertices position. All weight's bias effecting a single vertex's
                        // position must add up to 1.
                    }
                }

                //*** Calculate vertex normals using normal averaging ***///
                std::vector<XMFLOAT3> tempNormal;
                std::vector<XMFLOAT3> tempTangent;
                std::vector<XMFLOAT3> tempBitangent;

                // normalized and unnormalized normals
                XMFLOAT3 unnormalized = XMFLOAT3(0.0f, 0.0f, 0.0f);

                // Used to get vectors (sides) from the position of the verts
                float edge1X, edge1Y, edge1Z;
                float edge2X, edge2Y, edge2Z;

                float deltaUV1X, deltaUV1Y, deltaUV2X, deltaUV2Y;
                float tangentX, tangentY, tangentZ, bitangentX, bitangentY, bitangentZ, f;

                // Two edges of our triangle
                XMVECTOR edge1 = XMVectorSet(0.0f, 0.0f, 0.0f, 0.0f);
                XMVECTOR edge2 = XMVectorSet(0.0f, 0.0f, 0.0f, 0.0f);

                // Compute face normals
                for (int i = 0; i < subset.numTriangles; ++i) {
                    // Get the vector describing one edge of our triangle (edge 0,2)
                    edge2X = subset.vertices[subset.indices[(i * 3) + 2]].pos.x - subset.vertices[subset.indices[(i * 3)]].pos.x;
                    edge2Y = subset.vertices[subset.indices[(i * 3) + 2]].pos.y - subset.vertices[subset.indices[(i * 3)]].pos.y;
                    edge2Z = subset.vertices[subset.indices[(i * 3) + 2]].pos.z - subset.vertices[subset.indices[(i * 3)]].pos.z;
                    edge2 = XMVectorSet(edge2X, edge2Y, edge2Z, 0.0f);  // Create our first edge

                    // Get the vector describing another edge of our triangle (edge 2,1)
                    edge1X =
                        subset.vertices[subset.indices[(i * 3) + 1]].pos.x - subset.vertices[subset.indices[(i * 3)]].pos.x;
                    edge1Y =
                        subset.vertices[subset.indices[(i * 3) + 1]].pos.y - subset.vertices[subset.indices[(i * 3)]].pos.y;
                    edge1Z =
                        subset.vertices[subset.indices[(i * 3) + 1]].pos.z - subset.vertices[subset.indices[(i * 3)]].pos.z;
                    edge1 = XMVectorSet(edge1X, edge1Y, edge1Z, 0.0f);  // Create our second edge

                    // Cross multiply the two edge vectors to get the un-normalized face normal
                    XMStoreFloat3(&unnormalized, XMVector3Cross(edge1, edge2));

                    tempNormal.push_back(unnormalized);

                    // Tangent & Bitangent calculation
                    deltaUV1X = subset.vertices[subset.indices[(i * 3) + 1]].texCoord.x -
                                subset.vertices[subset.indices[(i * 3)]].texCoord.x;
                    deltaUV1Y = subset.vertices[subset.indices[(i * 3) + 1]].texCoord.y -
                                subset.vertices[subset.indices[(i * 3)]].texCoord.y;
                    deltaUV2X = subset.vertices[subset.indices[(i * 3) + 2]].texCoord.x -
                                subset.vertices[subset.indices[(i * 3)]].texCoord.x;
                    deltaUV2Y = subset.vertices[subset.indices[(i * 3) + 2]].texCoord.y -
                                subset.vertices[subset.indices[(i * 3)]].texCoord.y;

                    f = 1.0f / (deltaUV1X * deltaUV2Y - deltaUV2X * deltaUV1Y);

                    tangentX = f * (deltaUV2Y * edge1X - deltaUV1Y * edge2X);
                    tangentY = f * (deltaUV2Y * edge1Y - deltaUV1Y * edge2Y);
                    tangentZ = f * (deltaUV2Y * edge1Z - deltaUV1Y * edge2Z);

                    tempTangent.emplace_back(tangentX, tangentY, tangentZ);

                    bitangentX = f * (-deltaUV2X * edge1X + deltaUV1X * edge2X);
                    bitangentY = f * (-deltaUV2X * edge1Y + deltaUV1X * edge2Y);
                    bitangentZ = f * (-deltaUV2X * edge1Z + deltaUV1X * edge2Z);

                    tempBitangent.emplace_back(bitangentX, bitangentY, bitangentZ);
                }

                // Compute vertex normals (normal Averaging)
                XMFLOAT3 normalSum{0.0f, 0.0f, 0.0f};
                XMFLOAT3 tangentSum{0.0f, 0.0f, 0.0f};
                XMFLOAT3 bitangentSum{0.0f, 0.0f, 0.0f};

                // Go through each vertex
                for (int i = 0; i < subset.vertices.size(); ++i) {
                    // Check which triangles use this vertex
                    for (int j = 0; j < subset.numTriangles; ++j) {
                        if (subset.indices[j * 3] == i || subset.indices[(j * 3) + 1] == i || subset.indices[(j * 3) + 2] == i) {
                            normalSum.x += tempNormal[j].x;
                            normalSum.y += tempNormal[j].y;
                            normalSum.z += tempNormal[j].z;

                            tangentSum.x += tempTangent[j].x;
                            tangentSum.y += tempTangent[j].y;
                            tangentSum.z += tempTangent[j].z;

                            bitangentSum.x += tempBitangent[j].x;
                            bitangentSum.y += tempBitangent[j].y;
                            bitangentSum.z += tempBitangent[j].z;
                        }
                    }

                    // Normalize the normalSum vector
                    { 
                        XMVECTOR temp = XMVector3Normalize(XMVectorSet(normalSum.x, normalSum.y, normalSum.z, 0.0f));
                        XMStoreFloat3(&normalSum, temp);

                        temp = XMVector3Normalize(XMVectorSet(tangentSum.x, tangentSum.y, tangentSum.z, 0.0f));
                        XMStoreFloat3(&tangentSum, temp);

                        temp = XMVector3Normalize(XMVectorSet(bitangentSum.x, bitangentSum.y, bitangentSum.z, 0.0f));
                        XMStoreFloat3(&bitangentSum, temp);
                    }

                    // Store the normal and tangent in our current vertex
                    subset.vertices[i].normal.x = -normalSum.x;
                    subset.vertices[i].normal.y = -normalSum.y;
                    subset.vertices[i].normal.z = -normalSum.z;

                    subset.vertices[i].tangent.x = -tangentSum.x;
                    subset.vertices[i].tangent.y = -tangentSum.y;
                    subset.vertices[i].tangent.z = -tangentSum.z;

                    subset.vertices[i].biTangent.x = -bitangentSum.x;
                    subset.vertices[i].biTangent.y = -bitangentSum.y;
                    subset.vertices[i].biTangent.z = -bitangentSum.z;

                    ///////////////**************new**************////////////////////
                    // Create the joint space normal for easy normal calculations in animation
                    Vertex& tempVert = subset.vertices[i];                  // Get the current vertex
                    subset.jointSpaceNormals.push_back(XMFLOAT3(0, 0, 0));  // Push back a blank normal
                    XMVECTOR normal = XMVectorSet(0.0f, 0.0f, 0.0f, 0.0f);
                    XMVECTOR tangent = XMVectorSet(0.0f, 0.0f, 0.0f, 0.0f);
                    XMVECTOR bitnagent = XMVectorSet(0.0f, 0.0f, 0.0f, 0.0f);

                    for (int k = 0; k < tempVert.WeightCount; k++)  // Loop through each of the vertices weights
                    {
                        Joint& tempJoint =
                            mMD5Model.joints[subset.weights[tempVert.StartWeight + k].jointID];  // Get the joints orientation
                        XMVECTOR jointOrientation = XMVectorSet(tempJoint.orientation.x, tempJoint.orientation.y,
                                                                tempJoint.orientation.z, tempJoint.orientation.w);

                        // Calculate normal based off joints orientation (turn into joint space)
                        normal =
                            XMQuaternionMultiply(XMQuaternionMultiply(XMQuaternionInverse(jointOrientation),
                                                                      XMVectorSet(normalSum.x, normalSum.y, normalSum.z, 0.0f)),
                                                      jointOrientation);

                        XMStoreFloat3(&subset.weights[tempVert.StartWeight + k].normal,
                                      XMVector3Normalize(normal));  // Store the normalized quaternion into our weights normal

                        tangent = XMQuaternionMultiply(
                            XMQuaternionMultiply(XMQuaternionInverse(jointOrientation),
                                                 XMVectorSet(tangentSum.x, tangentSum.y, tangentSum.z, 0.0f)),
                                                      jointOrientation);

                        XMStoreFloat3(&subset.weights[tempVert.StartWeight + k].tangent,
                                      XMVector3Normalize(tangent));

                        bitnagent = XMQuaternionMultiply(
                            XMQuaternionMultiply(XMQuaternionInverse(jointOrientation),
                                                 XMVectorSet(bitangentSum.x, bitangentSum.y, bitangentSum.z, 0.0f)),
                                                      jointOrientation);

                        XMStoreFloat3(&subset.weights[tempVert.StartWeight + k].bitangent,
                                      XMVector3Normalize(bitnagent));
                    }
                }
                /// uploading verts & indices into CPU\GPU shared memory
                const int32_t indicesSize = sizeof(uint32_t) * subset.numTriangles * 3;
                const int32_t verticesSize = sizeof(Vertex) * subset.vertices.size();
                const auto uploadHeapProperties = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD);
                const auto uploadBufferVertDesc = CD3DX12_RESOURCE_DESC::Buffer(indicesSize);
                device->CreateCommittedResource(&uploadHeapProperties, D3D12_HEAP_FLAG_NONE, &uploadBufferVertDesc,
                                                D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&subset.indicesBuffer));
                const auto uploadBufferIndDesc = CD3DX12_RESOURCE_DESC::Buffer(verticesSize);
                device->CreateCommittedResource(&uploadHeapProperties, D3D12_HEAP_FLAG_NONE, &uploadBufferIndDesc,
                                                D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&subset.verticesBuffer));

                // Create buffer views
                subset.verticesBufferView.BufferLocation = subset.verticesBuffer->GetGPUVirtualAddress();
                subset.verticesBufferView.SizeInBytes = verticesSize;
                subset.verticesBufferView.StrideInBytes = sizeof(Vertex);

                subset.indicesBufferView.BufferLocation = subset.indicesBuffer->GetGPUVirtualAddress();
                subset.indicesBufferView.SizeInBytes = indicesSize;
                subset.indicesBufferView.Format = DXGI_FORMAT_R32_UINT;

                void* data;
                subset.indicesBuffer->Map(0, nullptr, &data);
                memcpy(data, subset.indices.data(), indicesSize);
                subset.indicesBuffer->Unmap(0, nullptr);

                subset.verticesBuffer->Map(0, nullptr, &data);
                memcpy(data, subset.vertices.data(), verticesSize);
                subset.verticesBuffer->Unmap(0, nullptr);
            }
        }
    } else {
        utils::log_err("Couldn't load animation file");
        return false;
    }

    return true;
}

void MD5Loader::Draw(ID3D12GraphicsCommandList* commandList) {
    assert(commandList);
    assert(mMD5Model.numSubsets > 0);
    for (int k = 0; k < mMD5Model.numSubsets; k++) {
            // Set the descriptor heap containing the texture srv
        ID3D12DescriptorHeap* heaps[] = {mMD5Model.subsets[k].texture.srvDescriptorHeap.Get()};
            commandList->SetDescriptorHeaps(1, heaps);
            // Set slot 0 of our root signature to point to our descriptor heap with
            // the texture SRV
            commandList->SetGraphicsRootDescriptorTable(
                0, mMD5Model.subsets[k].texture.srvDescriptorHeap->GetGPUDescriptorHandleForHeapStart());

        commandList->IASetVertexBuffers(0, 1, &mMD5Model.subsets[k].verticesBufferView);
        commandList->IASetIndexBuffer(&mMD5Model.subsets[k].indicesBufferView);
        commandList->DrawIndexedInstanced(mMD5Model.subsets[k].numTriangles * 3, 1, 0, 0, 0);
    }
}
