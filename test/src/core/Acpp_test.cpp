#include <iostream>

#include <gtest/gtest.h>
#include <sycl/sycl.hpp>

TEST(acpp_test, device_001){
    sycl::queue q;

    //std::cout << "PATH=" << std::getenv("PATH") << "\n";

    // std::print(
    //     "Device: {}\n USM has gpu: {}\n", 
    //     q.get_device().get_info<sycl::info::device::name>(),
    //     q.get_device().has(hipsycl::sycl::aspect::gpu)
    // );

    std::cout << "Device: " << q.get_device().get_info<sycl::info::device::name>() << std::endl
              << "USM has gpu: " << std::boolalpha << q.get_device().has(hipsycl::sycl::aspect::gpu) << std::endl;

    sycl::queue q2{sycl::gpu_selector{}};
}