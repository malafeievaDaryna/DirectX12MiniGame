#define FLASH_LIGHT_CALCULATION \
    "float getFlashLightAttenuation(float3 lightPos, float3 lightDir, float3 positionWorld, float radius)\n \
    {\n \
       float3 fragmentToLightDir = normalize(lightPos - positionWorld);\n \
       float cosAngle = saturate(dot(normalize(lightDir), normalize(positionWorld - lightPos)));\n \
       float lightFactor = 0.0;\n \
       if (cosAngle < 1 && cosAngle >= 0.99) lightFactor += smoothstep(0.99, 1, cosAngle); /* degree [0 : 9] */\n \
       if (cosAngle < 1 && cosAngle >= 0.956) lightFactor += smoothstep(0.956, 1, cosAngle); /* degrees [9 : 17] */\n \
       if (cosAngle < 0.985 && cosAngle >= 0.94) lightFactor += 1.0 - smoothstep(0.94, 0.985, cosAngle); /* degrees [10 : 19] */\n \
       if (cosAngle < 0.94 && cosAngle >= 0.92) lightFactor = max(smoothstep(0.92, 0.94, cosAngle), pow(0.38, 2)); /* degrees [19 : 22] */\n \
       if (cosAngle < 0.92) lightFactor = pow(0.38 * cosAngle / 0.92, 2); /* over 24 degrees */\n \
       float attenuation  = 1.0 - pow(saturate(length(lightPos - positionWorld) / radius), 3);\n \
       attenuation  =  lightFactor * attenuation;\n \
       return attenuation;\n \
    }\n"
namespace shaders {
namespace simple {
const char vs_shader[] =
    "cbuffer PerModelConstants : register (b0)\n"
    "{\n"
    "	matrix MVP;\n"
    "}\n"
    "struct VertexShaderOutput\n"
    "{\n"
    "	float4 position : SV_POSITION;\n"
    "	float2 uv : TEXCOORD;\n"
    "};\n"
    "VertexShaderOutput VS_main(\n"
    "	float3 position : POSITION,\n"
    "	float2 uv : TEXCOORD)\n"
    "{\n"
    "	VertexShaderOutput output;\n"
    "   output.position = mul(MVP, float4(position, 1));\n"
    "	output.uv = uv;\n"
    "	return output;\n"
    "}\n";
const char fs_shader[] =
    "Texture2D<float4> inputTexture : register(t0);\n"
    "SamplerState     texureSampler : register(s0);\n"
    "float4 PS_main (float4 position : SV_POSITION,\n"
    "				float2 uv : TEXCOORD) : SV_TARGET\n"
    "{\n"
    "	return inputTexture.Sample (texureSampler, uv);\n"
    "}\n";
}  // namespace simple
namespace directionalLight {
const char vs_shader[] =
    "cbuffer PerModelConstants : register (b0)\n"
    "{\n"
    "	matrix MVP;\n"
    "	matrix Model;\n"
    "	float4 LightPos;\n"
    "	float4 LightDir;\n"
    "}\n"
    "struct VertexShaderOutput\n"
    "{\n"
    "	float4 position : SV_POSITION;\n"
    "	float2 uv : TEXCOORD0;\n"
    "	float3 normal : NORMAL;\n"
    "	float3 positionWorld : TEXCOORD1;\n"
    "	float3 tangent : TANGENT;\n"
    "	float3 binormal : BINORMAL;\n"
    "};\n"
    "VertexShaderOutput VS_main(\n"
    "	float3 position : POSITION,\n"
    "	float2 uv : TEXCOORD,\n"
    "	float3 normal : NORMAL,\n"
    "	float3 tangent : TANGENT,\n"
    "	float3 bitangent : BITANGENT)\n"
    "{\n"
    "	VertexShaderOutput output;\n"
    "   float4 positionWorldSpace = mul(Model, float4(position, 1));"
    "   output.positionWorld = positionWorldSpace.xyz;"
    "   output.position = mul(MVP, float4(position, 1));\n"
    "	output.uv = uv;\n"
    "	float3x3 NormalMatrix = (float3x3)Model;\n"
    "	output.normal = normalize(mul(NormalMatrix, normal));\n"
    "	output.tangent = normalize(mul(NormalMatrix, tangent));\n"
    "	output.binormal = normalize(mul(NormalMatrix, bitangent));\n"
    "	return output;\n"
    "}\n";
const char fs_shader[] =
    "cbuffer PerModelConstants : register (b0)\n"
    "{\n"
    "	matrix MVP;\n"
    "	matrix Model;\n"
    "	float4 LightPos;\n"
    "	float4 LightDir;\n"
    "}\n" 
    "Texture2DArray<float4> inputTexture : register(t0);\n"
    "SamplerState     texureSampler : register(s0);\n" 
    FLASH_LIGHT_CALCULATION
    "float4 PS_main (float4 position : SV_POSITION,\n"
    "				float2 uv : TEXCOORD0, float3 normal : NORMAL, float3 positionWorld : TEXCOORD1, float3 tangent : TANGENT, float3 binormal : BINORMAL) : SV_TARGET\n"
    "{\n" 
    "	float3 specularMask = float3(1.0, 1.0, 1.0);\n" 
    "	float3 localNormal = normal;\n" 
    "	float width, height, texturesAmount;\n" 
    "	inputTexture.GetDimensions(width, height, texturesAmount);\n" 
    "   if (texturesAmount > 1) {\n" 
    "       float3 bump_normal = inputTexture.Sample (texureSampler, float3(uv[0], uv[1], 1)).xyz;\n" 
    "       bump_normal = (bump_normal * 2.0) - 1.0;\n" 
    "       localNormal = (bump_normal.x * tangent) + (bump_normal.y * binormal) + (bump_normal.z * normal);\n" 
    "   }\n" 
    "   if (texturesAmount > 2) {\n"
    "       specularMask = inputTexture.Sample (texureSampler, float3(uv[0], uv[1], 2)).xyz;\n"
    "   }\n" 
    "	float4 diffuseLighting = inputTexture.Sample (texureSampler, float3(uv[0], uv[1], 0));\n" 
    "	float3 fragmentToLightDir = normalize(LightPos.xyz - positionWorld);\n "
    "   float attenuation  = 0.4 * getFlashLightAttenuation(LightPos.xyz, LightDir.xyz, positionWorld, 1000.0f);\n"
    "   float3 viewDir  = fragmentToLightDir;"
    "   float3 reflectDir = reflect(-fragmentToLightDir, localNormal);"
    "   float3 halfwayDir = normalize(fragmentToLightDir + viewDir);"
    "   float specLightingPhong = pow(saturate(dot(viewDir, reflectDir)), 2.0);"
    "   float specLightingBlinn = pow(saturate(dot(localNormal, halfwayDir)), 1.0);"
    "   float specLighting = 1.2 * specLightingPhong + 0.25 * specLightingBlinn;"
    "	float3 specular = specLighting * specularMask;\n"
    "	float3 finalColor = attenuation * saturate(diffuseLighting.xyz * 0.6  + specular * 0.4);\n"
    "	return float4(finalColor.x, finalColor.y, finalColor.z, 1.0);\n"
    "}\n";
}  // namespace directionalLight
}  // namespace shaders
