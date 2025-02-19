namespace shaders {
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
    "	float4 position : POSITION,\n"
    "	float2 uv : TEXCOORD)\n"
    "{\n"
    "	VertexShaderOutput output;\n"
    "   output.position = mul(MVP, position);\n"
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
}
