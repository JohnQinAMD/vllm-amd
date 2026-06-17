// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#pragma once

#include <ATen/cuda/CUDAContext.h>
#include <string>

namespace vllm_mxfp8 {

// CDNA4 (gfx950) host guard. The decode MXFP8 kernels use the
// mfma_scale_f32_16x16x128_f8f6f4 path (compiled only under `#if __gfx950__`),
// so the device code is inert elsewhere; this backstops the Python-side
// supports_mx() gate against a direct op call on the wrong arch.
inline bool is_gfx950() {
  static const bool v =
      std::string(at::cuda::getCurrentDeviceProperties()->gcnArchName)
          .find("gfx950") != std::string::npos;
  return v;
}

}  // namespace vllm_mxfp8
