.visible .entry warpReduceSum(
    .param .u64 input,
    .param .u64 output
) {
    .reg .u32 t_idx;
    .reg .f32 val;

    mov.u32 t_idx, %tid.x;
    ld.global.f32 val, [input + t_idx * 4];

    // Warp shuffle reduction
    val = val + __shfl_down_sync(0xFFFFFFFF, val, 16);
    val = val + __shfl_down_sync(0xFFFFFFFF, val, 8);
    val = val + __shfl_down_sync(0xFFFFFFFF, val, 4);
    val = val + __shfl_down_sync(0xFFFFFFFF, val, 2);
    val = val + __shfl_down_sync(0xFFFFFFFF, val, 1);

    // Store reduced value
    st.global.f32 [output + t_idx * 4], val;
    ret;
}