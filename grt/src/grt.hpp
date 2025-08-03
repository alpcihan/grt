#pragma once

#define IMGUI_DEFINE_MATH_OPERATORS  // ImGUI ImVec maths

#include "common/vk_context.hpp"                    // Vulkan context creation
#include "imgui/imgui_axis.hpp"                     // Display of axis
#include "imgui/imgui_camera_widget.h"              // Camera UI
#include "imgui/imgui_helper.h"                     // Property editor
#include "nvh/primitives.hpp"                       // Various primitives
#include "nvvk/acceleration_structures.hpp"         // BLAS & TLAS creation helper
#include "nvvk/descriptorsets_vk.hpp"               // Descriptor set creation helper
#include "nvvk/extensions_vk.hpp"                   // Vulkan extension declaration
#include "nvvk/sbtwrapper_vk.hpp"                   // Shading binding table creation helper
#include "nvvk/shaders_vk.hpp"                      // Shader module creation wrapper
#include "nvvkhl/element_benchmark_parameters.hpp"  // For benchmark and tests
#include "nvvkhl/element_camera.hpp"                // To manipulate the camera
#include "nvvkhl/element_gui.hpp"                   // Application Menu / titlebar
#include "nvvkhl/gbuffer.hpp"                       // G-Buffer creation helper
#include "nvvkhl/pipeline_container.hpp"            // Container to hold pipelines
#include "nvvkhl/sky.hpp"                           // Sun & Sky
#include "nvvk/renderpasses_vk.hpp"

#include "grt_model.hpp"

#include <glm/gtx/quaternion.hpp>

namespace DH {
using namespace glm;
#include "shaders/device_host.h"  // Shared between host and device
#include "shaders/dh_bindings.h"  // Local device/host shared structures
}  // namespace DH

// Local shaders
#include "_autogen/raytrace_slang.h"

uint32_t MAXRAYRECURSIONDEPTH = 10;

class GRT : public nvvkhl::IAppElement
{
public:
  GRT(): m_model("/home/alp/Desktop/grt/grt/src/_data/data.bin", true){}
  ~GRT() override = default;

  void onAttach(nvvkhl::Application* app) override
  {
    nvh::ScopedTimer st(__FUNCTION__);

    m_app    = app;
    m_device = m_app->getDevice();

    m_dutil = std::make_unique<nvvk::DebugUtil>(m_device);  // Debug utility
    m_alloc = std::make_unique<nvvk::ResourceAllocatorDma>(m_device, m_app->getPhysicalDevice());
    m_rtSet = std::make_unique<nvvk::DescriptorSetContainer>(m_device);

    // Requesting ray tracing properties
    VkPhysicalDeviceProperties2 prop2{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, .pNext = &m_rtProperties};
    vkGetPhysicalDeviceProperties2(m_app->getPhysicalDevice(), &prop2);
    MAXRAYRECURSIONDEPTH = std::min(m_rtProperties.maxRayRecursionDepth, MAXRAYRECURSIONDEPTH);

    // Create utilities to create BLAS/TLAS and the Shading Binding Table (SBT)
    const uint32_t gct_queue_index = m_app->getQueue(0).familyIndex;
    m_sbt.setup(m_device, gct_queue_index, m_alloc.get(), m_rtProperties);

    // Create resources
    createScene();
    createVkBuffers();
    createBottomLevelAS();
    createTopLevelAS();
    createRtxPipeline();
  }

  void onDetach() override
  {
    vkDeviceWaitIdle(m_device);
    destroyResources();
  }

  void onResize(uint32_t width, uint32_t height) override
  {
    nvh::ScopedTimer st(__FUNCTION__);
    m_gBuffers = std::make_unique<nvvkhl::GBuffer>(m_device, m_alloc.get(), VkExtent2D{width, height}, m_colorFormat);
    writeRtDesc();
  }

  void onUIRender() override
  {
    {  // Setting menu
      ImGui::Begin("Settings");
      ImGuiH::CameraWidget();

      using namespace ImGuiH;
      PropertyEditor::begin();
      PropertyEditor::entry("Metallic", [&] { return ImGui::SliderFloat("#1", &m_pushConst.metallic, 0.0F, 1.0F); });
      PropertyEditor::entry("Roughness", [&] { return ImGui::SliderFloat("#1", &m_pushConst.roughness, 0.0F, 1.0F); });
      PropertyEditor::entry("Intensity", [&] { return ImGui::SliderFloat("#1", &m_pushConst.intensity, 0.0F, 10.0F); });
      PropertyEditor::entry("Depth",
                            [&] { return ImGui::SliderInt("#1", &m_pushConst.maxDepth, 0, MAXRAYRECURSIONDEPTH); });
      PropertyEditor::end();
      ImGui::Separator();
      ImGui::Text("Sun Orientation");
      PropertyEditor::begin();
      nvvkhl::skyParametersUI(m_skyParams);
      PropertyEditor::end();
      ImGui::End();
    }

    {  // Rendering Viewport
      ImGui::Begin("Viewport");

      // Display the G-Buffer image
      ImGui::Image(m_gBuffers->getDescriptorSet(), ImGui::GetContentRegionAvail());

      {  // Display orientation axis at the bottom left corner of the window
        float  axisSize = 25.F;
        ImVec2 pos      = ImGui::GetWindowPos();
        pos.y += ImGui::GetWindowSize().y;
        pos += ImVec2(axisSize * 1.1F, -axisSize * 1.1F) * ImGui::GetWindowDpiScale();  // Offset
        ImGuiH::Axis(pos, CameraManip.getMatrix(), axisSize);
      }

      ImGui::End();
    }
  }

  void memoryBarrier(VkCommandBuffer cmd)
  {
    VkMemoryBarrier mb{
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
    };
    VkPipelineStageFlags srcDstStage{VK_PIPELINE_STAGE_ALL_COMMANDS_BIT};
    vkCmdPipelineBarrier(cmd, srcDstStage, srcDstStage, 0, 1, &mb, 0, nullptr, 0, nullptr);
  }

  void onRender(VkCommandBuffer cmd) override
  {
    const nvvk::DebugUtil::ScopedCmdLabel sdbg = m_dutil->DBG_SCOPE(cmd);

    // Camera matrices
    glm::mat4 proj = glm::perspectiveRH_ZO(glm::radians(CameraManip.getFov()), CameraManip.getAspectRatio(),
                                           CameraManip.getClipPlanes().x, CameraManip.getClipPlanes().y);
    proj[1][1] *= -1;  // Vulkan has it inverted

    DH::FrameInfo finfo{.projInv = glm::inverse(proj), .viewInv = glm::inverse(CameraManip.getMatrix())};
    vkCmdUpdateBuffer(cmd, m_bFrameInfo.buffer, 0, sizeof(DH::FrameInfo), &finfo);  // Update FrameInfo
    vkCmdUpdateBuffer(cmd, m_bSkyParams.buffer, 0, sizeof(nvvkhl_shaders::SimpleSkyParameters), &m_skyParams);  // Update the sky
    memoryBarrier(cmd);  // Make sure the data has moved to device before rendering

    // Ray trace
    std::vector<VkDescriptorSet> desc_sets{m_rtSet->getSet()};
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, m_rtPipe.plines[0]);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, m_rtPipe.layout, 0, (uint32_t)desc_sets.size(),
                            desc_sets.data(), 0, nullptr);
    vkCmdPushConstants(cmd, m_rtPipe.layout, VK_SHADER_STAGE_ALL, 0, sizeof(DH::PushConstant), &m_pushConst);

    const std::array<VkStridedDeviceAddressRegionKHR, 4>& bindingTables = m_sbt.getRegions();
    const VkExtent2D&                                     size          = m_app->getViewportSize();
    vkCmdTraceRaysKHR(cmd, &bindingTables[0], &bindingTables[1], &bindingTables[2], &bindingTables[3], size.width, size.height, 1);

    // Read back data from arbitrary buffer (after rendering)
    VkMemoryBarrier readBarrier{
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_HOST_READ_BIT,
    };
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &readBarrier, 0, nullptr, 0, nullptr);
    readArbitraryBuffer();
  }

private:
  void readArbitraryBuffer()
  {
    // Map and read the buffer data
    float* bufferData = reinterpret_cast<float*>(m_alloc->map(m_bArbitraryBuffer));
    if (bufferData)
    {
      // Print first few values for demonstration
      printf("Arbitrary buffer values: ");
      for (int i = 0; i < 10 && i < 64; ++i)
      {
        printf("%.2f ", bufferData[i]);
      }
      printf("\n");
      m_alloc->unmap(m_bArbitraryBuffer);
    }
  }

  void createScene()
  {
    nvh::ScopedTimer st(__FUNCTION__);
    // No mesh/instance creation needed. Camera and material defaults only.
    CameraManip.setClipPlanes({0.1F, 100.0F});
    CameraManip.setLookat({0.0F, 0.0F, 3.0F}, {0.0F, 0.0F, -1.0F}, {0.0F, 1.0F, 0.0F});
    m_pushConst.intensity = 5.0F;
    m_pushConst.maxDepth  = 1;
    m_pushConst.roughness = 0.2F;
    m_pushConst.metallic  = 0.3F;
    m_skyParams = nvvkhl_shaders::initSimpleSkyParameters();
  }

  void createVkBuffers()
  {
    nvh::ScopedTimer st(__FUNCTION__);
    VkCommandBuffer cmd = m_app->createTempCmdBuffer();
    // Convert m_model.vertices (glm::vec3) to nvh::PrimitiveVertex
    std::vector<nvh::PrimitiveVertex> vertices;
    vertices.reserve(m_model.vertices.size());
    for(const auto& v : m_model.vertices) {
      nvh::PrimitiveVertex pv;
      pv.p = v;
      pv.n = glm::vec3(0.0f); // No normals
      pv.t = glm::vec2(0.0f); // No texcoords
      vertices.push_back(pv);
    }
    // Convert m_model.triangles (glm::ivec3) to nvh::PrimitiveTriangle
    std::vector<nvh::PrimitiveTriangle> triangles;
    triangles.reserve(m_model.triangles.size());
    for(const auto& t : m_model.triangles) {
      nvh::PrimitiveTriangle pt;
      pt.v = glm::uvec3(t.x, t.y, t.z);
      triangles.push_back(pt);
    }
    // Store as a single mesh
    m_meshes.clear();
    nvh::PrimitiveMesh mesh;
    mesh.vertices = std::move(vertices);
    mesh.triangles = std::move(triangles);
    m_meshes.push_back(std::move(mesh));
    m_bMeshes.resize(1);
    const VkBufferUsageFlags rt_usage_flag = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
                                             | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;
    m_bMeshes[0].vertices = m_alloc->createBuffer(cmd, m_meshes[0].vertices, rt_usage_flag);
    m_bMeshes[0].indices  = m_alloc->createBuffer(cmd, m_meshes[0].triangles, rt_usage_flag);
    m_dutil->DBG_NAME_IDX(m_bMeshes[0].vertices.buffer, 0);
    m_dutil->DBG_NAME_IDX(m_bMeshes[0].indices.buffer, 0);
    // Frame/sky/other buffers unchanged
    m_bFrameInfo = m_alloc->createBuffer(sizeof(DH::FrameInfo), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    m_dutil->DBG_NAME(m_bFrameInfo.buffer);
    m_bSkyParams = m_alloc->createBuffer(sizeof(nvvkhl_shaders::SimpleSkyParameters), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    m_dutil->DBG_NAME(m_bSkyParams.buffer);
    // No per-instance info needed, but keep a dummy InstanceInfo for descriptor compatibility
    std::vector<DH::InstanceInfo> inst_info(1);
    inst_info[0].transform = glm::mat4(1.0f);
    inst_info[0].materialID = 0;

    m_bInstInfoBuffer = m_alloc->createBuffer(cmd, inst_info, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
    m_dutil->DBG_NAME(m_bInstInfoBuffer.buffer);

    m_bAlbedos = m_alloc->createBuffer(cmd, m_model.albedos, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
    m_dutil->DBG_NAME(m_bAlbedos.buffer);

    m_bSHCoeffs = m_alloc->createBuffer(cmd, m_model.speculars, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
    m_dutil->DBG_NAME(m_bSHCoeffs.buffer);

    m_bPositions = m_alloc->createBuffer(cmd, m_model.positions, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
    m_dutil->DBG_NAME(m_bPositions.buffer);

    m_bRotations = m_alloc->createBuffer(cmd, m_model.rotations, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
    m_dutil->DBG_NAME(m_bRotations.buffer);

    m_bScales = m_alloc->createBuffer(cmd, m_model.scales, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
    m_dutil->DBG_NAME(m_bScales.buffer);

    m_bDensities = m_alloc->createBuffer(cmd, m_model.densities, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
    m_dutil->DBG_NAME(m_bDensities.buffer);

    // Create arbitrary buffer (64 floats) with host-visible memory for readback
    std::vector<float> arbitraryData(64, 0.0f);  // Initialize with zeros
    m_bArbitraryBuffer = m_alloc->createBuffer(cmd, arbitraryData, 
                                               VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                                               VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    m_dutil->DBG_NAME(m_bArbitraryBuffer.buffer);

    m_app->submitAndWaitTempCmdBuffer(cmd);
  }

  //--------------------------------------------------------------------------------------------------
  // Converting a PrimitiveMesh as input for BLAS
  //
  nvvk::AccelerationStructureGeometryInfo primitiveToGeometry(const nvh::PrimitiveMesh& prim, VkDeviceAddress vertexAddress, VkDeviceAddress indexAddress)
  {
    nvvk::AccelerationStructureGeometryInfo result;
    const auto                              triangleCount = static_cast<uint32_t>(prim.triangles.size());

    // Describe buffer as array of VertexObj.
    VkAccelerationStructureGeometryTrianglesDataKHR triangles{
        .sType        = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR,
        .vertexFormat = VK_FORMAT_R32G32B32_SFLOAT,  // vec3 vertex position data
        .vertexData   = {.deviceAddress = vertexAddress},
        .vertexStride = sizeof(nvh::PrimitiveVertex),
        .maxVertex    = static_cast<uint32_t>(prim.vertices.size()) - 1,
        .indexType    = VK_INDEX_TYPE_UINT32,
        .indexData    = {.deviceAddress = indexAddress},
    };

    // Identify the above data as containing opaque triangles.
    result.geometry = VkAccelerationStructureGeometryKHR{
        .sType        = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR,
        .geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR,
        .geometry     = {.triangles = triangles},
        .flags        = 0,
    };

    result.rangeInfo = VkAccelerationStructureBuildRangeInfoKHR{
        .primitiveCount  = triangleCount,
        .primitiveOffset = 0,
        .firstVertex     = 0,
        .transformOffset = 0,
    };

    return result;
  }

  //--------------------------------------------------------------------------------------------------
  // Create all bottom level acceleration structures (BLAS)
  //
  void createBottomLevelAS()
  {
    nvh::ScopedTimer st(__FUNCTION__);
    // Only one mesh
    m_blas.resize(1);
    std::vector<nvvk::AccelerationStructureBuildData> blasBuildData;
    blasBuildData.reserve(1);
    nvvk::AccelerationStructureBuildData buildData{VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR};
    const VkDeviceAddress vertexBufferAddress = m_bMeshes[0].vertices.address;
    const VkDeviceAddress indexBufferAddress  = m_bMeshes[0].indices.address;
    auto geo = primitiveToGeometry(m_meshes[0], vertexBufferAddress, indexBufferAddress);
    buildData.addGeometry(geo);
    buildData.finalizeGeometry(m_device, VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR |
                                                   VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_COMPACTION_BIT_KHR);
    blasBuildData.emplace_back(buildData);
    nvvk::BlasBuilder blasBuilder(m_alloc.get(), m_device);
    VkDeviceSize hintScratchBudget = 2'000'000;
    VkDeviceSize scratchSize = blasBuilder.getScratchSize(hintScratchBudget, blasBuildData);
    nvvk::Buffer scratchBuffer = m_alloc->createBuffer(scratchSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
    std::vector<VkDeviceAddress> scratchAddresses;
    blasBuilder.getScratchAddresses(hintScratchBudget, blasBuildData, scratchBuffer.address, scratchAddresses);
    bool finished = false;
    do {
      {
        VkCommandBuffer cmd = m_app->createTempCmdBuffer();
        finished = blasBuilder.cmdCreateParallelBlas(cmd, blasBuildData, m_blas, scratchAddresses, hintScratchBudget);
        m_app->submitAndWaitTempCmdBuffer(cmd);
      }
      {
        VkCommandBuffer cmd = m_app->createTempCmdBuffer();
        blasBuilder.cmdCompactBlas(cmd, blasBuildData, m_blas);
        m_app->submitAndWaitTempCmdBuffer(cmd);
        blasBuilder.destroyNonCompactedBlas();
      }
    } while(!finished);
    m_alloc->destroy(scratchBuffer);
  }

  //--------------------------------------------------------------------------------------------------
  // Create the top level acceleration structures, referencing all BLAS
  //
  void createTopLevelAS()
  {
    nvh::ScopedTimer st(__FUNCTION__);
    // Only one instance, identity transform
    std::vector<VkAccelerationStructureInstanceKHR> tlasInstances(1);
    tlasInstances[0].transform = nvvk::toTransformMatrixKHR(glm::mat4(1.0f));
    tlasInstances[0].instanceCustomIndex = 0;
    tlasInstances[0].mask = 0xFF;
    tlasInstances[0].instanceShaderBindingTableRecordOffset = 0;
    tlasInstances[0].flags = 0;
    tlasInstances[0].accelerationStructureReference = m_blas[0].address;
    VkCommandBuffer cmd = m_app->createTempCmdBuffer();
    nvvk::Buffer instancesBuffer = m_alloc->createBuffer(cmd, tlasInstances,
                                                         VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
                                                         VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
    nvvk::accelerationStructureBarrier(cmd, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR);
    nvvk::AccelerationStructureBuildData tlasBuildData{VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR};
    nvvk::AccelerationStructureGeometryInfo geometryInfo =
        tlasBuildData.makeInstanceGeometry(tlasInstances.size(), instancesBuffer.address);
    tlasBuildData.addGeometry(geometryInfo);
    auto sizeInfo = tlasBuildData.finalizeGeometry(m_device, VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR);
    nvvk::Buffer scratchBuffer = m_alloc->createBuffer(sizeInfo.buildScratchSize, VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                                                                                  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    m_tlas = m_alloc->createAcceleration(tlasBuildData.makeCreateInfo());
    tlasBuildData.cmdBuildAccelerationStructure(cmd, m_tlas.accel, scratchBuffer.address);
    m_app->submitAndWaitTempCmdBuffer(cmd);
    m_alloc->destroy(scratchBuffer);
    m_alloc->destroy(instancesBuffer);
    m_alloc->finalizeAndReleaseStaging();
  }

  //--------------------------------------------------------------------------------------------------
  // Pipeline for the ray tracer: all shaders, raygen, chit, anyhit, miss
  //
  void createRtxPipeline()
  {
    nvh::ScopedTimer st(__FUNCTION__);

    nvvkhl::PipelineContainer& p = m_rtPipe;
    p.plines.resize(1);

    // This descriptor set, holds the top level acceleration structure and the output image
    // Create Binding Set
    m_rtSet->addBinding(B_tlas, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1, VK_SHADER_STAGE_ALL);
    m_rtSet->addBinding(B_outImage, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_ALL);
    m_rtSet->addBinding(B_frameInfo, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_ALL);
    m_rtSet->addBinding(B_sceneDesc, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_ALL);
    m_rtSet->addBinding(B_skyParam, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_ALL);
    m_rtSet->addBinding(B_albedos, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_ALL);
    m_rtSet->addBinding(B_shCoeffs, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_ALL);
    m_rtSet->addBinding(B_positions, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_ALL);
    m_rtSet->addBinding(B_rotations, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_ALL);
    m_rtSet->addBinding(B_scales, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_ALL);
    m_rtSet->addBinding(B_densities, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_ALL);
    m_rtSet->addBinding(B_instances, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_ALL);
    m_rtSet->addBinding(B_arbitraryBuffer, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_ALL);
    m_rtSet->addBinding(B_vertex, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, (uint32_t)m_bMeshes.size(), VK_SHADER_STAGE_ALL);
    m_rtSet->addBinding(B_index, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, (uint32_t)m_bMeshes.size(), VK_SHADER_STAGE_ALL);
    m_rtSet->initLayout();
    m_rtSet->initPool(1);

    m_dutil->DBG_NAME(m_rtSet->getLayout());
    m_dutil->DBG_NAME(m_rtSet->getSet(0));

    // Creating all shaders
    enum StageIndices
    {
      eRaygen,
      eMiss,
      eClosestHit,
      eAnyHit,
      eShaderGroupCount
    };
    std::array<VkPipelineShaderStageCreateInfo, eShaderGroupCount> stages{};
    for(auto& s : stages)
      s.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;

    VkShaderModule shaderModule = nvvk::createShaderModule(m_device, &raytraceSlang[0], sizeof(raytraceSlang));
    stages[eRaygen].module      = shaderModule;
    stages[eRaygen].pName       = "rgenMain";
    stages[eRaygen].stage       = VK_SHADER_STAGE_RAYGEN_BIT_KHR;
    stages[eMiss].module        = shaderModule;
    stages[eMiss].pName         = "rmissMain";
    stages[eMiss].stage         = VK_SHADER_STAGE_MISS_BIT_KHR;
    stages[eClosestHit].module  = shaderModule;
    stages[eClosestHit].pName   = "rchitMain";
    stages[eClosestHit].stage   = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
    stages[eAnyHit].module      = shaderModule;
    stages[eAnyHit].pName       = "rahitMain";
    stages[eAnyHit].stage       = VK_SHADER_STAGE_ANY_HIT_BIT_KHR;

    m_dutil->setObjectName(stages[eRaygen].module, "Raygen");
    m_dutil->setObjectName(stages[eMiss].module, "Miss");
    m_dutil->setObjectName(stages[eClosestHit].module, "Closest Hit");
    m_dutil->setObjectName(stages[eAnyHit].module, "Any Hit");

    // Shader groups
    VkRayTracingShaderGroupCreateInfoKHR group{.sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR,
                                               .generalShader      = VK_SHADER_UNUSED_KHR,
                                               .closestHitShader   = VK_SHADER_UNUSED_KHR,
                                               .anyHitShader       = VK_SHADER_UNUSED_KHR,
                                               .intersectionShader = VK_SHADER_UNUSED_KHR};

    std::vector<VkRayTracingShaderGroupCreateInfoKHR> shader_groups;
    // Raygen
    group.type          = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
    group.generalShader = eRaygen;
    shader_groups.push_back(group);

    // Miss
    group.type          = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
    group.generalShader = eMiss;
    shader_groups.push_back(group);

    // closest hit + any hit group
    group.type             = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR;
    group.generalShader    = VK_SHADER_UNUSED_KHR;
    group.closestHitShader = eClosestHit;
    group.anyHitShader     = eAnyHit;
    shader_groups.push_back(group);

    // Push constant: we want to be able to update constants used by the shaders
    const VkPushConstantRange push_constant{VK_SHADER_STAGE_ALL, 0, sizeof(DH::PushConstant)};

    // Descriptor sets: one specific to ray tracing, and one shared with the rasterization pipeline
    std::vector<VkDescriptorSetLayout> rt_desc_set_layouts = {m_rtSet->getLayout()};  // , m_pContainer[eGraphic].dstLayout};
    VkPipelineLayoutCreateInfo pipeline_layout_create_info{
        .sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount         = static_cast<uint32_t>(rt_desc_set_layouts.size()),
        .pSetLayouts            = rt_desc_set_layouts.data(),
        .pushConstantRangeCount = 1,
        .pPushConstantRanges    = &push_constant,
    };
    NVVK_CHECK(vkCreatePipelineLayout(m_device, &pipeline_layout_create_info, nullptr, &p.layout));
    m_dutil->DBG_NAME(p.layout);

    // Assemble the shader stages and recursion depth info into the ray tracing pipeline
    VkRayTracingPipelineCreateInfoKHR ray_pipeline_info{
        .sType                        = VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR,
        .stageCount                   = static_cast<uint32_t>(stages.size()),  // Stages are shader
        .pStages                      = stages.data(),
        .groupCount                   = static_cast<uint32_t>(shader_groups.size()),
        .pGroups                      = shader_groups.data(),
        .maxPipelineRayRecursionDepth = MAXRAYRECURSIONDEPTH,  // Ray dept
        .layout                       = p.layout,
    };
    NVVK_CHECK(vkCreateRayTracingPipelinesKHR(m_device, {}, {}, 1, &ray_pipeline_info, nullptr, &p.plines[0]));
    m_dutil->DBG_NAME(p.plines[0]);

    // Creating the SBT
    m_sbt.create(p.plines[0], ray_pipeline_info);

    // Removing temp modules
#if USE_SLANG
    vkDestroyShaderModule(m_device, shaderModule, nullptr);
#else
    for(const VkPipelineShaderStageCreateInfo& s : stages)
      vkDestroyShaderModule(m_device, s.module, nullptr);
#endif
  }

  void writeRtDesc()
  {
    // Write to descriptors
    VkAccelerationStructureKHR tlas = m_tlas.accel;
    VkWriteDescriptorSetAccelerationStructureKHR desc_as_info{.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR,
                                                              .accelerationStructureCount = 1,
                                                              .pAccelerationStructures    = &tlas};
    const VkDescriptorImageInfo  image_info{{}, m_gBuffers->getColorImageView(), VK_IMAGE_LAYOUT_GENERAL};
    const VkDescriptorBufferInfo dbi_unif{m_bFrameInfo.buffer, 0, VK_WHOLE_SIZE};
    const VkDescriptorBufferInfo dbi_sky{m_bSkyParams.buffer, 0, VK_WHOLE_SIZE};
    const VkDescriptorBufferInfo albedos_desc{m_bAlbedos.buffer, 0, VK_WHOLE_SIZE};
    const VkDescriptorBufferInfo shCoeffs_desc{m_bSHCoeffs.buffer, 0, VK_WHOLE_SIZE};
    const VkDescriptorBufferInfo positions_desc{m_bPositions.buffer, 0, VK_WHOLE_SIZE};
    const VkDescriptorBufferInfo rotations_desc{m_bRotations.buffer, 0, VK_WHOLE_SIZE};
    const VkDescriptorBufferInfo scales_desc{m_bScales.buffer, 0, VK_WHOLE_SIZE};
    const VkDescriptorBufferInfo densities_desc{m_bDensities.buffer, 0, VK_WHOLE_SIZE};
    const VkDescriptorBufferInfo inst_desc{m_bInstInfoBuffer.buffer, 0, VK_WHOLE_SIZE};
    const VkDescriptorBufferInfo arbitrary_desc{m_bArbitraryBuffer.buffer, 0, VK_WHOLE_SIZE};

    std::vector<VkDescriptorBufferInfo> vertex_desc;
    std::vector<VkDescriptorBufferInfo> index_desc;
    vertex_desc.reserve(m_bMeshes.size());
    index_desc.reserve(m_bMeshes.size());
    for(auto& m : m_bMeshes)
    {
      vertex_desc.push_back({m.vertices.buffer, 0, VK_WHOLE_SIZE});
      index_desc.push_back({m.indices.buffer, 0, VK_WHOLE_SIZE});
    }

    std::vector<VkWriteDescriptorSet> writes;
    writes.emplace_back(m_rtSet->makeWrite(0, B_tlas, &desc_as_info));
    writes.emplace_back(m_rtSet->makeWrite(0, B_outImage, &image_info));
    writes.emplace_back(m_rtSet->makeWrite(0, B_frameInfo, &dbi_unif));
    writes.emplace_back(m_rtSet->makeWrite(0, B_skyParam, &dbi_sky));
    writes.emplace_back(m_rtSet->makeWrite(0, B_albedos, &albedos_desc));
    writes.emplace_back(m_rtSet->makeWrite(0, B_shCoeffs, &shCoeffs_desc));
    writes.emplace_back(m_rtSet->makeWrite(0, B_positions, &positions_desc));
    writes.emplace_back(m_rtSet->makeWrite(0, B_rotations, &rotations_desc));
    writes.emplace_back(m_rtSet->makeWrite(0, B_scales, &scales_desc));
    writes.emplace_back(m_rtSet->makeWrite(0, B_densities, &densities_desc));
    writes.emplace_back(m_rtSet->makeWrite(0, B_instances, &inst_desc));
    writes.emplace_back(m_rtSet->makeWrite(0, B_arbitraryBuffer, &arbitrary_desc));
    writes.emplace_back(m_rtSet->makeWriteArray(0, B_vertex, vertex_desc.data()));
    writes.emplace_back(m_rtSet->makeWriteArray(0, B_index, index_desc.data()));

    vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
  }

  void destroyResources()
  {
    for(PrimitiveMeshVk& m : m_bMeshes)
    {
      m_alloc->destroy(m.vertices);
      m_alloc->destroy(m.indices);
    }
    m_alloc->destroy(m_bFrameInfo);
    m_alloc->destroy(m_bInstInfoBuffer);
    m_alloc->destroy(m_bAlbedos);
    m_alloc->destroy(m_bSHCoeffs);
    m_alloc->destroy(m_bDensities);
    m_alloc->destroy(m_bSkyParams);
    m_alloc->destroy(m_bArbitraryBuffer);

    m_rtSet->deinit();
    m_gBuffers.reset();

    m_rtPipe.destroy(m_device);
    m_sbt.destroy();

    for(auto& b : m_blas)
      m_alloc->destroy(b);
    m_alloc->destroy(m_tlas);
  }

  void onLastHeadlessFrame() override
  {
    m_app->saveImageToFile(m_gBuffers->getColorImage(), m_gBuffers->getSize(),
                           nvh::getExecutablePath().replace_extension(".jpg").string(), 95);
  }

private:
  nvvkhl::Application*                          m_app = nullptr;
  std::unique_ptr<nvvk::DebugUtil>              m_dutil;
  std::unique_ptr<nvvk::ResourceAllocatorDma>   m_alloc;
  std::unique_ptr<nvvk::DescriptorSetContainer> m_rtSet;  // Descriptor set

  VkFormat                            m_colorFormat = VK_FORMAT_R8G8B8A8_UNORM;  // Color format of the image
  VkDevice                            m_device      = VK_NULL_HANDLE;            // Convenient
  std::unique_ptr<nvvkhl::GBuffer>    m_gBuffers;                                // G-Buffers: color + depth
  nvvkhl_shaders::SimpleSkyParameters m_skyParams{};

  // Resources
  struct PrimitiveMeshVk
  {
    nvvk::Buffer vertices;
    nvvk::Buffer indices;
  };
  std::vector<PrimitiveMeshVk> m_bMeshes;  // Each primitive holds a buffer of vertices and indices
  nvvk::Buffer                 m_bFrameInfo;
  nvvk::Buffer                 m_bInstInfoBuffer;
  nvvk::Buffer                 m_bAlbedos;
  nvvk::Buffer                 m_bSHCoeffs;
  nvvk::Buffer                 m_bPositions;
  nvvk::Buffer                 m_bRotations;
  nvvk::Buffer                 m_bScales;
  nvvk::Buffer                 m_bDensities;
  nvvk::Buffer                 m_bSkyParams;
  nvvk::Buffer                 m_bArbitraryBuffer;

  std::vector<nvvk::AccelKHR> m_blas;  // Bottom-level AS
  nvvk::AccelKHR              m_tlas;  // Top-level AS

  // Data and setting
  std::vector<nvh::PrimitiveMesh> m_meshes;

  // Pipeline
  DH::PushConstant m_pushConst{};  // Information sent to the shader

  VkPhysicalDeviceRayTracingPipelinePropertiesKHR m_rtProperties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR};
  nvvk::SBTWrapper          m_sbt;     // Shading binding table wrapper
  nvvkhl::PipelineContainer m_rtPipe;  // Hold pipelines and layout

  // Model
  GRTModel m_model;
};