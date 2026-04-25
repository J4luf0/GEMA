#include <vector>

#include <sycl/sycl.hpp>

#include "TensorParallel.hpp"
#include "MemoryBackendUSM.hpp"

namespace gema{
   
    
    template <class T>
    TensorParallel<T>::TensorParallel(const std::vector<uint64_t>& newTensorDimensionSizes) 
    : Tensor<T>(newTensorDimensionSizes) {
        this->tensor_ = LinearContainer<T>(MemoryBackendUSM<T, sycl::usm::alloc::device>(&queue_));
    }
}