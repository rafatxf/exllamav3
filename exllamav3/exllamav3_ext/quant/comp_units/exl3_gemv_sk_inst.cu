#include <cuda_fp16.h>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;
#include "../../util.h"
#include "../../util.cuh"
#include "../exl3_gemv_sk_kernel.cuh"
#include "../exl3_gemv.cuh"

// Instances of the sm_75 split-k GEMV (exl3_gemv_sk_kernel.cuh), 4 tiles per warp. m = 1: activation-as-A MMAs,
// prefetch depth 2; 2 <= m <= 8: weights-as-A MMAs, depth 1 (measured best at m = 1, 4 and 8)
void* exl3_gemv_sk_select_kernel(int bits, int cb, bool c_fp32, int mmode)
{
    #define SEL(bits_, cb_, fp32_) \
        if (bits == bits_ && cb == cb_ && c_fp32 == fp32_) \
            return mmode == 0 ? (void*) exl3_gemv_sk_kernel<bits_, fp32_, cb_, 0, EXL3_GEMV_SK_WNT, EXL3_GEMV_SK_PF(0), false> \
                              : (void*) exl3_gemv_sk_kernel<bits_, fp32_, cb_, 1, EXL3_GEMV_SK_WNT, EXL3_GEMV_SK_PF(1), true>;
    #define SEL2(bits_, cb_) SEL(bits_, cb_, false) SEL(bits_, cb_, true)
    SEL2(2, 1) SEL2(2, 2)
    SEL2(3, 1) SEL2(3, 2)
    SEL2(4, 0) SEL2(4, 1) SEL2(4, 2)
    SEL2(5, 0) SEL2(5, 1) SEL2(5, 2)
    SEL2(6, 0) SEL2(6, 1) SEL2(6, 2)
    SEL2(8, 0) SEL2(8, 1) SEL2(8, 2)
    #undef SEL2
    #undef SEL
    return nullptr;
}
