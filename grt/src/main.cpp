#include <memory>
#include "grt.hpp"

int main(int argc, char** argv)
{
  nvvkhl::ApplicationCreateInfo appInfo;

  nvh::CommandLineParser cli(PROJECT_NAME);
  cli.addArgument({"--headless"}, &appInfo.headless, "Run in headless mode");
  cli.addArgument({"--frames"}, &appInfo.headlessFrameCount, "Number of frames to render in headless mode");
  cli.parse(argc, argv);


  VkPhysicalDeviceAccelerationStructureFeaturesKHR accel_feature{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR};
  VkPhysicalDeviceRayTracingPipelineFeaturesKHR rt_pipeline_feature{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR};

  // Configure Vulkan context creation
  VkContextSettings vkSetup;
  if(!appInfo.headless)
  {
    nvvkhl::addSurfaceExtensions(vkSetup.instanceExtensions);
    vkSetup.deviceExtensions.push_back({VK_KHR_SWAPCHAIN_EXTENSION_NAME});
  }
  vkSetup.instanceExtensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
  vkSetup.deviceExtensions.push_back({VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME});
  vkSetup.deviceExtensions.push_back({VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME, &accel_feature});  // To build acceleration structures
  vkSetup.deviceExtensions.push_back({VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME, &rt_pipeline_feature});  // To use vkCmdTraceRaysKHR
  vkSetup.deviceExtensions.push_back({VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME});  // Required by ray tracing pipeline
  vkSetup.deviceExtensions.push_back({VK_KHR_DYNAMIC_RENDERING_EXTENSION_NAME});  // Require for Undockable Viewport

#if USE_HLSL || USE_SLANG  // DXC is automatically adding the extension
  VkPhysicalDeviceRayQueryFeaturesKHR rayqueryFeature{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR};
  vkSetup.deviceExtensions.push_back({VK_KHR_RAY_QUERY_EXTENSION_NAME, &rayqueryFeature});
#endif  // USE_HLSL

#if(VK_HEADER_VERSION >= 283)
  // To enable ray tracing validation, set the NV_ALLOW_RAYTRACING_VALIDATION=1 environment variable
  // https://developer.nvidia.com/blog/ray-tracing-validation-at-the-driver-level/
  // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_NV_ray_tracing_validation.html
  VkPhysicalDeviceRayTracingValidationFeaturesNV validationFeatures = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_VALIDATION_FEATURES_NV};
  vkSetup.deviceExtensions.push_back({VK_NV_RAY_TRACING_VALIDATION_EXTENSION_NAME, &validationFeatures, false});
#endif
  
  // Create Vulkan context
  auto vkContext = std::make_unique<VulkanContext>(vkSetup);
  if(!vkContext->isValid())
    std::exit(0);

  // Loading the Vulkan extension pointers
  load_VK_EXTENSIONS(vkContext->getInstance(), vkGetInstanceProcAddr, vkContext->getDevice(), vkGetDeviceProcAddr);

  // Configure application creation
  appInfo.name                  = fmt::format("{} ({})", PROJECT_NAME, SHADER_LANGUAGE_STR);
  appInfo.vSync                 = true;
  appInfo.instance              = vkContext->getInstance();
  appInfo.device                = vkContext->getDevice();
  appInfo.physicalDevice        = vkContext->getPhysicalDevice();
  appInfo.queues                = vkContext->getQueueInfos();
  appInfo.hasUndockableViewport = true;

  // Create the application
  auto app = std::make_unique<nvvkhl::Application>(appInfo);

  // Create the test framework
  auto test = std::make_shared<nvvkhl::ElementBenchmarkParameters>(argc, argv);

#if(VK_HEADER_VERSION >= 283)
  // Check if ray tracing validation is supported
  if(validationFeatures.rayTracingValidation == VK_TRUE)
  {
    LOGI("Ray tracing validation supported");
  }
#endif

  // Add all application elements
  app->addElement(test);
  app->addElement(std::make_shared<nvvkhl::ElementCamera>());       // Camera manipulation
  app->addElement(std::make_shared<nvvkhl::ElementDefaultMenu>());  // Menu / Quit
  app->addElement(std::make_shared<nvvkhl::ElementDefaultWindowTitle>("", fmt::format("({})", SHADER_LANGUAGE_STR)));  // Window title info
  app->addElement(std::make_shared<GRT>());

  app->run();
  app.reset();
  vkContext.reset();

  return test->errorCode();
}
