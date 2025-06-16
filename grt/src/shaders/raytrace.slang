 #version 460

 #include "common/shaders/constants.hlsli"
 #include "common/shaders/ggx.hlsli"
 #include "common/shaders/sky.hlsli"
 #include "device_host.h"
 #include "dh_bindings.h"
 
 #define MISS_DEPTH 1000
 
 // Bindings
 [[vk::push_constant]] ConstantBuffer<PushConstant> pushConst;
 [[vk::binding(B_tlas)]] RaytracingAccelerationStructure topLevelAS;
 [[vk::binding(B_outImage)]] RWTexture2D<float4> outImage;
 [[vk::binding(B_frameInfo)]] ConstantBuffer<FrameInfo> frameInfo;
 [[vk::binding(B_skyParam)]] ConstantBuffer<SimpleSkyParameters> skyInfo;
 [[vk::binding(B_albedos)]] StructuredBuffer<float3> albedos;
 [[vk::binding(B_shCoeffs)]] StructuredBuffer<float3> shCoeffs;
 [[vk::binding(B_densities)]] StructuredBuffer<float> densities;
 [[vk::binding(B_instances)]] StructuredBuffer<InstanceInfo> instanceInfo;
 [[vk::binding(B_vertex)]] StructuredBuffer<Vertex> vertices[];
 [[vk::binding(B_index)]] StructuredBuffer<uint3> indices[];
 
 
 //-----------------------------------------------------------------------
 // Payload 
 // See: https://microsoft.github.io/DirectX-Specs/d3d/Raytracing.html#example
 struct HitPayload
 {
   float3 color;
   float depth;
   float transmittance;
 };
 
 
 //-----------------------------------------------------------------------
 // Hit state information
 struct HitState
 {
    int particleId;
    float distance;
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
   uint instanceID = InstanceIndex();
   uint meshID = InstanceID();
   uint primitiveID = PrimitiveIndex();

   // Material info
   float3 radiance = radianceFromSpH(3, instanceID, WorldRayDirection(), true);

   float density = densities[instanceID];
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