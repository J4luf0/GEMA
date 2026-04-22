#include <iostream>

#include <gtest/gtest.h>
#include <sycl/sycl.hpp>

TEST(acpp_test, device_001){
    sycl::queue q;

    //std::cout << "PATH=" << std::getenv("PATH") << "\n";

    std::cout << "Device: "
                << q.get_device().get_info<sycl::info::device::name>()
                << "\n";

    sycl::queue q2{sycl::gpu_selector{}};
}