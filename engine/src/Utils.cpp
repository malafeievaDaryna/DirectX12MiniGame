#include "Utils.h"

#include <WICTextureLoader.h>
#include <cassert>
#include <unordered_map>
#include "d3dx12.h"
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

namespace utils {
Texture2DResource CreateTexture(ID3D12Device* device, ID3D12GraphicsCommandList* uploadCommandList,
                                const std::string& textureFileName) {
    assert(uploadCommandList && device && !textureFileName.empty());
    std::unordered_map<std::string, Texture2DResource> texturesCache;

    if (auto search = texturesCache.find(textureFileName); search != texturesCache.end()) {
        log_debug("textures cache has such texture", textureFileName);
        return search->second;
    } else {
        log_debug("creation new texture for", textureFileName);
        auto& textureResource = texturesCache[textureFileName];
        int texWidth, texHeight, texChannels;
        std::size_t imageSizeTotal = 0u;
        using dataTexturetPtr = std::unique_ptr<stbi_uc, decltype(&stbi_image_free)>;

        std::string path = TEXTURE_PATH + textureFileName;

        /// STBI_rgb_alpha coerces to have ALPHA chanel for consistency with alphaless images
        stbi_uc* pixels = stbi_load(path.c_str(), &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);
        assert(pixels);
        dataTexturetPtr textureData(pixels, stbi_image_free);
        imageSizeTotal += texWidth * texHeight * 4LL;

        static const auto defaultHeapProperties = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT);
        const auto resourceDesc = CD3DX12_RESOURCE_DESC::Tex2D(DXGI_FORMAT_R8G8B8A8_UNORM_SRGB, texWidth, texHeight, 1, 1);

        device->CreateCommittedResource(&defaultHeapProperties, D3D12_HEAP_FLAG_NONE, &resourceDesc,
                                        D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&textureResource.image));

        static const auto uploadHeapProperties = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD);
        const auto uploadBufferSize = GetRequiredIntermediateSize(textureResource.image.Get(), 0, 1);
        const auto uploadBufferDesc = CD3DX12_RESOURCE_DESC::Buffer(uploadBufferSize);

        device->CreateCommittedResource(&uploadHeapProperties, D3D12_HEAP_FLAG_NONE, &uploadBufferDesc,
                                        D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&textureResource.stagingBuffer));

        D3D12_SUBRESOURCE_DATA srcData;
        srcData.pData = textureData.get();
        srcData.RowPitch = texWidth * 4;
        srcData.SlicePitch = texWidth * texHeight * 4;

        UpdateSubresources(uploadCommandList, textureResource.image.Get(), textureResource.stagingBuffer.Get(), 0, 0, 1,
                           &srcData);
        const auto transition = CD3DX12_RESOURCE_BARRIER::Transition(textureResource.image.Get(), D3D12_RESOURCE_STATE_COPY_DEST,
                                                                     D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
        uploadCommandList->ResourceBarrier(1, &transition);

        D3D12_SHADER_RESOURCE_VIEW_DESC shaderResourceViewDesc = {};
        shaderResourceViewDesc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
        shaderResourceViewDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        shaderResourceViewDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM_SRGB;
        shaderResourceViewDesc.Texture2D.MipLevels = 1;
        shaderResourceViewDesc.Texture2D.MostDetailedMip = 0;
        shaderResourceViewDesc.Texture2D.ResourceMinLODClamp = 0.0f;

        // We need one descriptor heap to store our texture SRV which cannot go
        // into the root signature. So create a SRV type heap with one entry
        D3D12_DESCRIPTOR_HEAP_DESC descriptorHeapDesc = {};
        descriptorHeapDesc.NumDescriptors = 1;
        // This heap contains SRV, UAV or CBVs -- in our case one SRV
        descriptorHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
        descriptorHeapDesc.NodeMask = 0;
        descriptorHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;

        device->CreateDescriptorHeap(&descriptorHeapDesc, IID_PPV_ARGS(&textureResource.srvDescriptorHeap));

        device->CreateShaderResourceView(textureResource.image.Get(), &shaderResourceViewDesc,
                                         textureResource.srvDescriptorHeap->GetCPUDescriptorHandleForHeapStart());
        return textureResource;
    }
}
// TODO alternative to stbi functional
// void CreateTextureWIC(ID3D12Device* device, ID3D12GraphicsCommandList* uploadCommandList, const std::string& textureFileName) {
//    DirectX::ResourceUploadBatch resourceUpload(device);
//    resourceUpload.Begin();
//    DirectX::CreateWICTextureFromFileEx(device, resourceUpload, L"textures\\texture.png", 0, D3D12_RESOURCE_FLAG_NONE,
//                                        DirectX::WIC_LOADER_FORCE_RGBA32 | DirectX::WIC_LOADER_MIP_AUTOGEN,
//                                        mImage.ReleaseAndGetAddressOf());
//    // Upload the resources to the GPU.
//    auto uploadResourcesFinished = resourceUpload.End(mCommandQueue.Get());
//    // Wait for the upload thread to terminate
//    uploadResourcesFinished.wait();
//}
}  // namespace utils