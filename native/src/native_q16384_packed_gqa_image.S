// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Approaching AI Authors

#ifndef AIMA_Q16384_PACKED_GQA_HSACO_PATH
#error "AIMA_Q16384_PACKED_GQA_HSACO_PATH must name the qualified HSACO"
#endif
#ifndef AIMA_Q8192_PACKED_GQA_HSACO_PATH
#error "AIMA_Q8192_PACKED_GQA_HSACO_PATH must name the qualified HSACO"
#endif

    .section .rodata.aima_q16384_packed_gqa,"a",@progbits
    .balign 16
    .global aima_q16384_packed_gqa_hsaco_start
    .type aima_q16384_packed_gqa_hsaco_start, @object
aima_q16384_packed_gqa_hsaco_start:
    .incbin AIMA_Q16384_PACKED_GQA_HSACO_PATH
    .global aima_q16384_packed_gqa_hsaco_end
    .type aima_q16384_packed_gqa_hsaco_end, @object
aima_q16384_packed_gqa_hsaco_end:
    .size aima_q16384_packed_gqa_hsaco_start, .-aima_q16384_packed_gqa_hsaco_start

    .balign 16
    .global aima_q8192_packed_gqa_hsaco_start
    .type aima_q8192_packed_gqa_hsaco_start, @object
aima_q8192_packed_gqa_hsaco_start:
    .incbin AIMA_Q8192_PACKED_GQA_HSACO_PATH
    .global aima_q8192_packed_gqa_hsaco_end
    .type aima_q8192_packed_gqa_hsaco_end, @object
aima_q8192_packed_gqa_hsaco_end:
    .size aima_q8192_packed_gqa_hsaco_start, .-aima_q8192_packed_gqa_hsaco_start

    .section .note.GNU-stack,"",@progbits
