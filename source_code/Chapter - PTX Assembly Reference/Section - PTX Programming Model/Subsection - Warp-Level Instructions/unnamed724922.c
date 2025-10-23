// PTX Code for Vector Addition
.visible .entry vectorAdd(
    .param .u64 A,
    .param .u64 B,
    .param .u64 C,
    .param .u32 N
) {
    .reg .u32 t_idx;
    .reg .f32 a, b, c;

    mov.u32 t_idx, %tid.x; // Get thread index
    mul.wide.u32 %rd1, t_idx, 4;
    add.u64 %rd2, A, %rd1;
    add.u64 %rd3, B, %rd1;
    add.u64 %rd4, C, %rd1;
    
    ld.global.f32 a, [%rd2]; // Load A[i]
    ld.global.f32 b, [%rd3]; // Load B[i]
    add.f32 c, a, b; // Perform addition
    st.global.f32 [%rd4], c; // Store result in C[i]

    ret;
}