 #version 460

 #include "common/shaders/constants.hlsli"
 #include "common/shaders/ggx.hlsli"
 #include "common/shaders/sky.hlsli"
 #include "device_host.h"
 #include "dh_bindings.h"
 
 #define MISS_DEPTH 1000
 #define MIN_TRANSMITTANCE 0.03
 #define KERNEL_DEGREE 4
 #define MAX_PARTICLE_ALPHA 0.99
 #define MIN_PARTICLE_ALPHA (1.0f/255.0f)
 #define MIN_PARTICLE_KERNEL_DENSITY 0.01
  
 // Bindings
 [[vk::push_constant]] ConstantBuffer<PushConstant> pushConst;
 [[vk::binding(B_tlas)]] RaytracingAccelerationStructure topLevelAS;
 [[vk::binding(B_outImage)]] RWTexture2D<float4> outImage;
 [[vk::binding(B_frameInfo)]] ConstantBuffer<FrameInfo> frameInfo;
 [[vk::binding(B_skyParam)]] ConstantBuffer<SimpleSkyParameters> skyInfo;
 [[vk::binding(B_albedos)]] StructuredBuffer<float3> albedos;
 [[vk::binding(B_shCoeffs)]] StructuredBuffer<float3> shCoeffs;
 [[vk::binding(B_positions)]] StructuredBuffer<float3> positions;
 [[vk::binding(B_rotations)]] StructuredBuffer<float4> rotations;
 [[vk::binding(B_scales)]] StructuredBuffer<float3> scales;
 [[vk::binding(B_densities)]] StructuredBuffer<float> densities;
 [[vk::binding(B_instances)]] StructuredBuffer<InstanceInfo> instanceInfo;

 //-----------------------------------------------------------------------
 // Payload 
 // See: https://microsoft.github.io/DirectX-Specs/d3d/Raytracing.html#example
 struct HitPayload
 {
    float3 color;
    float depth;
    float transmittance;
   
    float3 radiance;
    float density;
    float3 normal;
    float hitDistance;
    float rayLastHitDistance;
    float hitCount; // TODO (operel): convert to uint32
 };
 
 
 //-----------------------------------------------------------------------
 // Hit state information
 struct HitState
 {
    int particleId;
    float distance;
 };

 struct Parameters {
    float3 position;
    float3 scale;
    float3x3 rotationT;
    float density;
};
 
 //-----------------------------------------------------------------------
 // RAY GEN
 //-----------------------------------------------------------------------
 [shader("raygeneration")]
 void rgenMain()
 {
   float2 launchID = (float2)DispatchRaysIndex().xy;
   float2 launchSize = (float2)DispatchRaysDimensions().xy;

   const uint rayFlags = RAY_FLAG_CULL_BACK_FACING_TRIANGLES;
   const float2 clipCoords = (launchID + 0.5) / launchSize * 2.0 - 1.0;
   const float4 viewCoords = mul(frameInfo.projInv, float4(clipCoords, 1.0, 1.0));
 
   RayDesc ray;
   ray.Origin = mul(frameInfo.viewInv, float4(0.0, 0.0, 0.0, 1.0)).xyz;
   ray.Direction = mul(frameInfo.viewInv, float4(normalize(viewCoords.xyz), 0.0)).xyz;
 
   ray.TMin = 0.001;
   ray.TMax = INFINITE;
 
   // Initial state
   HitPayload payload;
   payload.color = float3(0, 0, 0);
   payload.depth = 0;
   payload.transmittance = 1.0;

   while (payload.transmittance > 0.1f && payload.depth < INFINITE) {
       TraceRay(topLevelAS, rayFlags, 0xff, 0, 0, 0, ray, payload);
       ray.Origin += ray.Direction * (payload.depth + 10e-9);
   }
  
   // -Wpayload-access-perf
   float depth = payload.depth;

   float3 color = payload.color;
   outImage[int2(launchID)] = float4(color, 1.0);
 }
 
 
 //-----------------------------------------------------------------------
 // CLOSEST HIT
 //-----------------------------------------------------------------------
 [shader("closesthit")]
 void rchitMain(inout HitPayload payload, in BuiltInTriangleIntersectionAttributes attr)
 {
   float3 barycentrics = float3(1 - attr.barycentrics.x - attr.barycentrics.y, attr.barycentrics.x, attr.barycentrics.y);
   uint meshID = InstanceID();
   uint primitiveID = PrimitiveIndex();

   uint particleIdx = primitiveID / 20; // TODO: This is a hack to get the particle index from the primitive ID.

   // Material info
   float3 radiance = radianceFromSpH(3, particleIdx, WorldRayDirection(), true);

   float density = densities[particleIdx];
   density = 0.25f;

   payload.color += payload.transmittance * radiance * density;
   payload.transmittance *= (1.0f - density);
   payload.depth = RayTCurrent();
 }

 //-----------------------------------------------------------------------
 // ANY HIT
 //-----------------------------------------------------------------------
 [shader("anyhit")]
 void rahitMain(inout HitPayload payload, in BuiltInTriangleIntersectionAttributes attr)
 {
 }
 
 //-----------------------------------------------------------------------
 // MISS 
 //-----------------------------------------------------------------------
 [shader("miss")]
 void rmissMain(inout HitPayload payload)
 {
   // float3 sky_color = evalSimpleSky(skyInfo, WorldRayDirection());
   // payload.color += sky_color * payload.weight;

   // payload.color = float3(0.0, 0.0, 0.0);
   // payload.depth = MISS_DEPTH; // Stop

   payload.depth = INFINITE;
 }

//-----------------------------------------------------------------------
// Response
//-----------------------------------------------------------------------
 float processHitFromBuffer<let backToFront : bool>(
    float3 rayOrigin,
    float3 rayDirection,
    uint32_t particleIdx,
    inout float transmittance,
    inout float integratedDepth)
{
    float alpha = 0.0f;
    float depth;
    float3 normal;
    float hitWeight = 0.0f;

    bool isHit = hit(
                    rayOrigin,
                    rayDirection,
                    Parameters(
                        positions[particleIdx],
                        scales[particleIdx],
                        rotationMatrixTranspose(rotations[particleIdx]),
                        densities[particleIdx]),
                    alpha,
                    depth);

    if (isHit)
    {
        hitWeight = integrateHit(
                                alpha, 
                                transmittance, 
                                depth, 
                                integratedDepth);
    }

    return hitWeight;
}

bool hit(
    float3 rayOrigin,
    float3 rayDirection,
    Parameters parameters,
    out float alpha,
    inout float depth) {

    float3 canonicalRayOrigin;
    float3 canonicalRayDirection;
    cannonicalRay(
        rayOrigin,
        rayDirection,
        parameters,
        canonicalRayOrigin,
        canonicalRayDirection);

    const float maxResponse = canonicalRayMaxKernelResponse(
        canonicalRayOrigin,
        canonicalRayDirection);

    alpha = min(MAX_PARTICLE_ALPHA, maxResponse * parameters.density); // needed for the stability of the backward pass
    const bool acceptHit = ((maxResponse > MIN_PARTICLE_KERNEL_DENSITY) && (alpha > MIN_PARTICLE_ALPHA));
    if (acceptHit)
    {
        depth = canonicalRayDistance(canonicalRayOrigin, canonicalRayDirection, parameters.scale);
    }
    return acceptHit;
}

void cannonicalRay(
    in float3 rayOrigin,
    in float3 rayDirection,
    in Parameters parameters,
    out float3 particleRayOrigin,
    out float3 particleRayDirection, ) {
    const float3 giscl  = float3(1.0f) / parameters.scale;
    const float3 gposc  = (rayOrigin - parameters.position);
    const float3 gposcr = mul(parameters.rotationT, gposc);
    particleRayOrigin   = giscl * gposcr;

    const float3 rayDirR = mul(parameters.rotationT, rayDirection);
    const float3 grdu    = giscl * rayDirR;
    particleRayDirection = normalize(grdu);
}

float canonicalRayMaxKernelResponse(
    float3 canonicalRayOrigin,
    float3 canonicalRayDirection) {
    const float grayDist = canonicalRayMinSquaredDistance(canonicalRayOrigin, canonicalRayDirection);

    /// generalized gaussian of degree n : scaling is s = -4.5/3^n
    switch (KERNEL_DEGREE) {
    case 8: // Zenzizenzizenzic
    {
        /*static const*/ float s = -0.000685871056241;
        const float grayDistSq   = grayDist * grayDist;
        return exp(s * grayDistSq * grayDistSq);
    }
    case 5: // Quintic
    {
        /*static const*/ float s = -0.0185185185185;
        return exp(s * grayDist * grayDist * sqrt(grayDist));
    }
    case 4: // Tesseractic
    {
        /*static const*/ float s = -0.0555555555556;
        return exp(s * grayDist * grayDist);
    }
    case 3: // Cubic
    {
        /*static const*/ float s = -0.166666666667;
        return exp(s * grayDist * sqrt(grayDist));
    }
    case 1: // Laplacian
    {
        /*static const*/ float s = -1.5f;
        return exp(s * sqrt(grayDist));
    }
    case 0: // Linear
    {
        /* static const */ float s = -0.329630334487;
        return max(1 + s * sqrt(grayDist), 0.f);
    }
    default: // Quadratic
    {
        /*static const*/ float s = -0.5f;
        return exp(s * grayDist);
    }
    }
}

float canonicalRayDistance(
    float3 canonicalRayOrigin,
    float3 canonicalRayDirection,
    float3 scale) {
    const float3 grds = scale * canonicalRayDirection * dot(canonicalRayDirection, -1 * canonicalRayOrigin);
    return sqrt(dot(grds, grds));
}

float canonicalRayMinSquaredDistance(
    float3 canonicalRayOrigin,
    float3 canonicalRayDirection) {
    const float3 gcrod = cross(canonicalRayDirection, canonicalRayOrigin);
    return dot(gcrod, gcrod);
}

float integrateHit(
    in float alpha,
    inout float transmittance,
    in float depth,
    inout float integratedDepth)
{
    const float weight = alpha * transmittance;
    integratedDepth += depth * weight;
    transmittance *= (1 - alpha);
    return weight; 
}

float3x3 rotationMatrixTranspose(float4 quaternion)
{
    // quaternion represented as (r,x,y,z)
    const float xx = quaternion.y * quaternion.y;
    const float yy = quaternion.z * quaternion.z;
    const float zz = quaternion.w * quaternion.w;
    const float xy = quaternion.y * quaternion.z;
    const float xz = quaternion.y * quaternion.w;
    const float yz = quaternion.z * quaternion.w;
    const float rx = quaternion.x * quaternion.y;
    const float ry = quaternion.x * quaternion.z;
    const float rz = quaternion.x * quaternion.w;

    // Compute rotation matrix from quaternion
    return float3x3(float3((1.f - 2.f * (yy + zz)), 2.f * (xy + rz), 2.f * (xz - ry)),
                    float3(2.f * (xy - rz), (1.f - 2.f * (xx + zz)), 2.f * (yz + rx)),
                    float3(2.f * (xz + ry), 2.f * (yz - rx), (1.f - 2.f * (xx + yy))));
}

 //-----------------------------------------------------------------------
 // Spherical Harmonics
 //-----------------------------------------------------------------------
 static const float SH_C0 = 0.28209479177387814;
 static const float SH_C1 = 0.4886025119029199;

 static const float SH_C2[5] = {
     1.0925484305920792,
     -1.0925484305920792,
     0.31539156525252005,
     -1.0925484305920792,
     0.5462742152960396
 };

 static const float SH_C3[7] = {
     -0.5900435899266435,
     2.890611442640554,
     -0.4570457994644658,
     0.3731763325901154,
     -0.4570457994644658,
     1.445305721320277,
     -0.5900435899266435
 };

 float3 radianceFromSpH(int deg, int instanceID, float3 rdir, bool clamped = true)
 {
     float3 rad = SH_C0 * albedos[instanceID];

     if (deg > 0)
     {
         float x = rdir.x;
         float y = rdir.y;
         float z = rdir.z;

         rad = rad
            - SH_C1 * y * shCoeffs[instanceID]
            + SH_C1 * z * shCoeffs[instanceID + 1]
            - SH_C1 * x * shCoeffs[instanceID + 2];

         if (deg > 1)
         {
             float xx = x * x, yy = y * y, zz = z * z;
             float xy = x * y, yz = y * z, xz = x * z;

             rad = rad
                + SH_C2[0] * xy * shCoeffs[instanceID + 3]
                + SH_C2[1] * yz * shCoeffs[instanceID + 4]
                + SH_C2[2] * (2.0 * zz - xx - yy) * shCoeffs[instanceID + 5]
                + SH_C2[3] * xz * shCoeffs[instanceID + 6]
                + SH_C2[4] * (xx - yy) * shCoeffs[instanceID + 7];

             if (deg > 2)
             {
                 rad = rad
                    + SH_C3[0] * y * (3.0 * xx - yy) * shCoeffs[instanceID + 8]
                    + SH_C3[1] * xy * z * shCoeffs[instanceID + 9]
                    + SH_C3[2] * y * (4.0 * zz - xx - yy) * shCoeffs[instanceID + 10]
                    + SH_C3[3] * z * (2.0 * zz - 3.0 * xx - 3.0 * yy) * shCoeffs[instanceID + 11]
                    + SH_C3[4] * x * (4.0 * zz - xx - yy) * shCoeffs[instanceID + 12]
                    + SH_C3[5] * z * (xx - yy) * shCoeffs[instanceID + 13]
                    + SH_C3[6] * x * (xx - 3.0 * yy) * shCoeffs[instanceID + 14];
             }
         }
     }

     rad += float3(0.5, 0.5, 0.5);

     return clamped ? max(rad, float3(0.0, 0.0, 0.0)) : rad;
 } 