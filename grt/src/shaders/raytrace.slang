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
   float3 albedo = albedos[instanceID].xyz;
   float density = densities[instanceID];

   payload.color += payload.transmittance * albedo * density;
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
 
 