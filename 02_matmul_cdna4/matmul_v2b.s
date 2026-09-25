	.amdgcn_target "amdgcn-amd-amdhsa-unknown-gfx950"
	.amdhsa_code_object_version 6
	.text
	.globl	matmul_v2b
	.p2align	8
	.type	matmul_v2b,@function
matmul_v2b:
	s_load_dwordx2 s[8:9], s[0:1], 0x0
	s_load_dwordx4 s[4:7], s[0:1], 0x10
	s_load_dwordx2 s[10:11], s[0:1], 0x28
	s_load_dwordx2 s[16:17], s[0:1], 0x48
	s_lshl_b32 s14, s3, 8
	s_ashr_i32 s3, s14, 31
	s_waitcnt lgkmcnt(0)
	s_mul_i32 s3, s4, s3
	s_mul_hi_u32 s15, s4, s14
	s_add_i32 s3, s15, s3
	s_mul_i32 s5, s5, s14
	v_readfirstlane_b32 s18, v0
	s_add_i32 s5, s3, s5
	s_mul_i32 s4, s4, s14
	s_lshr_b32 s19, s18, 6
	s_bfe_u32 s12, s18, 0x10006
	s_and_b32 s13, s18, 0x80
	s_lshl_b64 s[4:5], s[4:5], 1
	s_add_u32 s4, s8, s4
	s_addc_u32 s5, s9, s5
	s_lshl_b32 s15, s2, 7
	s_ashr_i32 s2, s15, 31
	s_mul_i32 s2, s10, s2
	s_mul_hi_u32 s3, s10, s15
	s_add_i32 s2, s3, s2
	s_mul_i32 s3, s11, s15
	s_add_i32 s3, s2, s3
	s_mul_i32 s2, s10, s15
	s_lshl_b64 s[2:3], s[2:3], 1
	s_add_u32 s8, s6, s2
	v_lshrrev_b32_e32 v1, 3, v0
	s_addc_u32 s2, s7, s3
	v_xor_b32_e32 v2, v1, v0
	s_and_b32 s9, s2, 0xffff
	v_lshlrev_b32_e32 v2, 4, v2
	v_lshlrev_b32_e32 v1, 15, v1
	s_movk_i32 s2, 0x70
	v_and_or_b32 v129, v2, s2, v1
	v_and_b32_e32 v130, 15, v0
	v_bfe_u32 v2, v0, 4, 2
	s_lshl_b32 s26, s12, 13
	v_and_b32_e32 v128, 63, v0
	v_or_b32_e32 v1, s13, v130
	v_and_b32_e32 v3, 7, v0
	v_bitop3_b32 v0, v2, v0, 7 bitop3:0x78
	v_lshl_or_b32 v4, v130, 7, s26
	v_or_b32_e32 v4, 0x8000, v4
	v_lshlrev_b32_e32 v0, 4, v0
	v_lshlrev_b32_e32 v1, 7, v1
	s_sub_i32 s3, s16, s14
	s_and_b32 s18, s18, 0xc0
	v_or_b32_e32 v131, v4, v0
	v_or_b32_e32 v132, v1, v0
	v_bitop3_b32 v0, v2, v3, 4 bitop3:0x36
	s_lshl_b32 s6, s3, 15
	s_mov_b32 s7, 0x27000
	s_sub_i32 s3, s17, s15
	s_lshl_b32 s2, s19, 10
	s_lshl_b32 s21, s18, 4
	v_lshlrev_b32_e32 v0, 4, v0
	v_mov_b32_e32 v108, 0
	s_and_b32 s5, s5, 0xffff
	s_lshl_b32 s10, s3, 15
	s_mov_b32 s11, s7
	s_add_i32 s3, s2, 0x1000
	s_add_i32 s16, s2, 0x2000
	s_add_i32 s17, s2, 0x3000
	s_or_b32 s18, s21, 0x4000
	s_or_b32 s19, s21, 0x5000
	s_or_b32 s20, s21, 0x6000
	s_addk_i32 s21, 0x7000
	s_add_i32 s22, s2, 0x8000
	s_add_i32 s23, s2, 0x9000
	s_add_i32 s24, s2, 0xa000
	s_add_i32 s25, s2, 0xb000
	v_or_b32_e32 v133, v4, v0
	v_or_b32_e32 v134, v1, v0
	s_mov_b32 s26, 0
	v_mov_b32_e32 v109, v108
	v_mov_b32_e32 v110, v108
	v_mov_b32_e32 v111, v108
	v_mov_b32_e32 v120, v108
	v_mov_b32_e32 v121, v108
	v_mov_b32_e32 v122, v108
	v_mov_b32_e32 v123, v108
	v_mov_b32_e32 v116, v108
	v_mov_b32_e32 v117, v108
	v_mov_b32_e32 v118, v108
	v_mov_b32_e32 v119, v108
	v_mov_b32_e32 v112, v108
	v_mov_b32_e32 v113, v108
	v_mov_b32_e32 v114, v108
	v_mov_b32_e32 v115, v108
	v_mov_b32_e32 v104, v108
	v_mov_b32_e32 v105, v108
	v_mov_b32_e32 v106, v108
	v_mov_b32_e32 v107, v108
	v_mov_b32_e32 v100, v108
	v_mov_b32_e32 v101, v108
	v_mov_b32_e32 v102, v108
	v_mov_b32_e32 v103, v108
	v_mov_b32_e32 v96, v108
	v_mov_b32_e32 v97, v108
	v_mov_b32_e32 v98, v108
	v_mov_b32_e32 v99, v108
	v_mov_b32_e32 v92, v108
	v_mov_b32_e32 v93, v108
	v_mov_b32_e32 v94, v108
	v_mov_b32_e32 v95, v108
	v_mov_b32_e32 v88, v108
	v_mov_b32_e32 v89, v108
	v_mov_b32_e32 v90, v108
	v_mov_b32_e32 v91, v108
	v_mov_b32_e32 v84, v108
	v_mov_b32_e32 v85, v108
	v_mov_b32_e32 v86, v108
	v_mov_b32_e32 v87, v108
	v_mov_b32_e32 v80, v108
	v_mov_b32_e32 v81, v108
	v_mov_b32_e32 v82, v108
	v_mov_b32_e32 v83, v108
	v_mov_b32_e32 v76, v108
	v_mov_b32_e32 v77, v108
	v_mov_b32_e32 v78, v108
	v_mov_b32_e32 v79, v108
	v_mov_b32_e32 v72, v108
	v_mov_b32_e32 v73, v108
	v_mov_b32_e32 v74, v108
	v_mov_b32_e32 v75, v108
	v_mov_b32_e32 v68, v108
	v_mov_b32_e32 v69, v108
	v_mov_b32_e32 v70, v108
	v_mov_b32_e32 v71, v108
	v_mov_b32_e32 v64, v108
	v_mov_b32_e32 v65, v108
	v_mov_b32_e32 v66, v108
	v_mov_b32_e32 v67, v108
	v_mov_b32_e32 v60, v108
	v_mov_b32_e32 v61, v108
	v_mov_b32_e32 v62, v108
	v_mov_b32_e32 v63, v108
	v_mov_b32_e32 v56, v108
	v_mov_b32_e32 v57, v108
	v_mov_b32_e32 v58, v108
	v_mov_b32_e32 v59, v108
	v_mov_b32_e32 v52, v108
	v_mov_b32_e32 v53, v108
	v_mov_b32_e32 v54, v108
	v_mov_b32_e32 v55, v108
	v_mov_b32_e32 v48, v108
	v_mov_b32_e32 v49, v108
	v_mov_b32_e32 v50, v108
	v_mov_b32_e32 v51, v108
	v_mov_b32_e32 v44, v108
	v_mov_b32_e32 v45, v108
	v_mov_b32_e32 v46, v108
	v_mov_b32_e32 v47, v108
	v_mov_b32_e32 v40, v108
	v_mov_b32_e32 v41, v108
	v_mov_b32_e32 v42, v108
	v_mov_b32_e32 v43, v108
	v_mov_b32_e32 v36, v108
	v_mov_b32_e32 v37, v108
	v_mov_b32_e32 v38, v108
	v_mov_b32_e32 v39, v108
	v_mov_b32_e32 v32, v108
	v_mov_b32_e32 v33, v108
	v_mov_b32_e32 v34, v108
	v_mov_b32_e32 v35, v108
	v_mov_b32_e32 v28, v108
	v_mov_b32_e32 v29, v108
	v_mov_b32_e32 v30, v108
	v_mov_b32_e32 v31, v108
	v_mov_b32_e32 v24, v108
	v_mov_b32_e32 v25, v108
	v_mov_b32_e32 v26, v108
	v_mov_b32_e32 v27, v108
	v_mov_b32_e32 v20, v108
	v_mov_b32_e32 v21, v108
	v_mov_b32_e32 v22, v108
	v_mov_b32_e32 v23, v108
	v_mov_b32_e32 v16, v108
	v_mov_b32_e32 v17, v108
	v_mov_b32_e32 v18, v108
	v_mov_b32_e32 v19, v108
	v_mov_b32_e32 v12, v108
	v_mov_b32_e32 v13, v108
	v_mov_b32_e32 v14, v108
	v_mov_b32_e32 v15, v108
	v_mov_b32_e32 v8, v108
	v_mov_b32_e32 v9, v108
	v_mov_b32_e32 v10, v108
	v_mov_b32_e32 v11, v108
	v_mov_b32_e32 v4, v108
	v_mov_b32_e32 v5, v108
	v_mov_b32_e32 v6, v108
	v_mov_b32_e32 v7, v108
	v_mov_b32_e32 v0, v108
	v_mov_b32_e32 v1, v108
	v_mov_b32_e32 v2, v108
	v_mov_b32_e32 v3, v108
	v_mov_b32_e32 v124, v108
	v_mov_b32_e32 v125, v108
	v_mov_b32_e32 v126, v108
	v_mov_b32_e32 v127, v108
.LBB0_1:
	s_mov_b32 m0, s2
	v_lshl_add_u32 v135, s26, 7, v129
	s_barrier
	buffer_load_dwordx4 v135, s[4:7], 0 offen lds
	v_or_b32_e32 v136, 0x100000, v135
	s_mov_b32 m0, s3
	v_or_b32_e32 v137, 0x200000, v135
	buffer_load_dwordx4 v136, s[4:7], 0 offen lds
	s_mov_b32 m0, s16
	v_or_b32_e32 v138, 0x300000, v135
	buffer_load_dwordx4 v137, s[4:7], 0 offen lds
	s_mov_b32 m0, s17
	v_or_b32_e32 v139, 0x400000, v135
	buffer_load_dwordx4 v138, s[4:7], 0 offen lds
	s_mov_b32 m0, s18
	s_nop 0
	buffer_load_dwordx4 v139, s[4:7], 0 offen lds
	v_or_b32_e32 v139, 0x500000, v135
	s_mov_b32 m0, s19
	s_nop 0
	buffer_load_dwordx4 v139, s[4:7], 0 offen lds
	v_or_b32_e32 v139, 0x600000, v135
	s_mov_b32 m0, s20
	s_nop 0
	buffer_load_dwordx4 v139, s[4:7], 0 offen lds
	v_or_b32_e32 v139, 0x700000, v135
	s_mov_b32 m0, s21
	s_nop 0
	buffer_load_dwordx4 v139, s[4:7], 0 offen lds
	s_mov_b32 m0, s22
	s_nop 0
	buffer_load_dwordx4 v135, s[8:11], 0 offen lds
	s_mov_b32 m0, s23
	s_nop 0
	buffer_load_dwordx4 v136, s[8:11], 0 offen lds
	s_mov_b32 m0, s24
	s_nop 0
	buffer_load_dwordx4 v137, s[8:11], 0 offen lds
	s_mov_b32 m0, s25
	s_nop 0
	buffer_load_dwordx4 v138, s[8:11], 0 offen lds
	s_waitcnt vmcnt(0)
	s_barrier

	; each wave computes A[128,64] x B[64,64]
	; each MFMA tile is A[16,32] x B[16,32]
	;   |----------|         |----------|
	;   | A0 16x32 |         | B0 16x32 |
	;   | A1 16x32 |         | B1 16x32 |
	;   |    ...   |         | B2 16x32 |
	;   | A7 16x32 |         | B3 16x32 |
	;   |----------|         |----------|

	; ##### inner k tile 0 #####
	; load B0-B3
	ds_read_b128 v[136:139], v131
	ds_read_b128 v[140:143], v131 offset:2048
	ds_read_b128 v[144:147], v131 offset:4096
	ds_read_b128 v[148:151], v131 offset:6144
	; load A0-A3
	ds_read_b128 v[152:155], v132
	ds_read_b128 v[156:159], v132 offset:2048
	ds_read_b128 v[160:163], v132 offset:4096
	ds_read_b128 v[164:167], v132 offset:6144

	; wait B0-B3 + A0, compute A0 x [B0, B1, B2, B3]
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_16x16x32_bf16 v[108:111], v[136:139], v[152:155], v[108:111]
	v_mfma_f32_16x16x32_bf16 v[120:123], v[140:143], v[152:155], v[120:123]
	v_mfma_f32_16x16x32_bf16 v[116:119], v[144:147], v[152:155], v[116:119]
	v_mfma_f32_16x16x32_bf16 v[112:115], v[148:151], v[152:155], v[112:115]

	; wait A1, compute A1 x [B0, B1, B2, B3]
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_16x16x32_bf16 v[104:107], v[136:139], v[156:159], v[104:107]
	v_mfma_f32_16x16x32_bf16 v[100:103], v[140:143], v[156:159], v[100:103]
	v_mfma_f32_16x16x32_bf16 v[96:99], v[144:147], v[156:159], v[96:99]
	v_mfma_f32_16x16x32_bf16 v[92:95], v[148:151], v[156:159], v[92:95]

	; wait A2, compute A2 x [B0, B1, B2, B3]
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_16x16x32_bf16 v[88:91], v[136:139], v[160:163], v[88:91]
	v_mfma_f32_16x16x32_bf16 v[84:87], v[140:143], v[160:163], v[84:87]
	v_mfma_f32_16x16x32_bf16 v[80:83], v[144:147], v[160:163], v[80:83]
	v_mfma_f32_16x16x32_bf16 v[76:79], v[148:151], v[160:163], v[76:79]

	; wait A3, compute A3 x [B0, B1, B2, B3]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x32_bf16 v[72:75], v[136:139], v[164:167], v[72:75]
	v_mfma_f32_16x16x32_bf16 v[68:71], v[140:143], v[164:167], v[68:71]
	v_mfma_f32_16x16x32_bf16 v[64:67], v[144:147], v[164:167], v[64:67]
	v_mfma_f32_16x16x32_bf16 v[60:63], v[148:151], v[164:167], v[60:63]

	; load A4-A7, reuse A0-A3 registers
	ds_read_b128 v[152:155], v132 offset:8192
	ds_read_b128 v[156:159], v132 offset:10240
	ds_read_b128 v[160:163], v132 offset:12288
	ds_read_b128 v[164:167], v132 offset:14336

	; wait A4, compute A4 x [B0, B1, B2, B3]
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_16x16x32_bf16 v[56:59], v[136:139], v[152:155], v[56:59]
	v_mfma_f32_16x16x32_bf16 v[52:55], v[140:143], v[152:155], v[52:55]
	v_mfma_f32_16x16x32_bf16 v[48:51], v[144:147], v[152:155], v[48:51]
	v_mfma_f32_16x16x32_bf16 v[44:47], v[148:151], v[152:155], v[44:47]

	; prefetch B3 of inner k tile 1, reuse A4 register
	ds_read_b128 v[152:155], v133 offset:6144

	; wait A5, compute A5 x [B0, B1, B2, B3]
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_16x16x32_bf16 v[40:43], v[136:139], v[156:159], v[40:43]
	v_mfma_f32_16x16x32_bf16 v[36:39], v[140:143], v[156:159], v[36:39]
	v_mfma_f32_16x16x32_bf16 v[32:35], v[144:147], v[156:159], v[32:35]
	v_mfma_f32_16x16x32_bf16 v[28:31], v[148:151], v[156:159], v[28:31]

	; wait A6, compute A6 x [B0, B1, B2, B3]
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_16x16x32_bf16 v[24:27], v[136:139], v[160:163], v[24:27]
	v_mfma_f32_16x16x32_bf16 v[20:23], v[140:143], v[160:163], v[20:23]
	v_mfma_f32_16x16x32_bf16 v[16:19], v[144:147], v[160:163], v[16:19]
	v_mfma_f32_16x16x32_bf16 v[12:15], v[148:151], v[160:163], v[12:15]

	; wait A7, compute A7 x B0
	; prefetch B0 of inner k tile 1, reuse B0 register
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_16x16x32_bf16 v[8:11], v[136:139], v[164:167], v[8:11]
	ds_read_b128 v[136:139], v133

	; compute A7 x B1
	; prefetch B1 of inner k tile 1, reuse B1 register
	v_mfma_f32_16x16x32_bf16 v[4:7], v[140:143], v[164:167], v[4:7]
	ds_read_b128 v[140:143], v133 offset:2048

	; compute A7 x B2
	; prefetch B2 of inner k tile 1, reuse B2 register
	v_mfma_f32_16x16x32_bf16 v[0:3], v[144:147], v[164:167], v[0:3]
	ds_read_b128 v[144:147], v133 offset:4096

	; compute A7 x B3
	v_mfma_f32_16x16x32_bf16 v[124:127], v[148:151], v[164:167], v[124:127]

	; ##### inner k tile 1 #####
	; all B have been prefetched previously
	; load A0, reuse B3 register
	ds_read_b128 v[148:151], v134
	; load A1-A3, reuse A5-A7 registers
	ds_read_b128 v[156:159], v134 offset:2048
	ds_read_b128 v[160:163], v134 offset:4096
	ds_read_b128 v[164:167], v134 offset:6144
	s_add_i32 s26, s26, 1

	; wait A0-A3, compute [A0, A1, A2, A3] x [B0, B1, B2, B3]
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_16x16x32_bf16 v[108:111], v[136:139], v[148:151], v[108:111]
	s_cmpk_lg_i32 s26, 0x100
	v_mfma_f32_16x16x32_bf16 v[120:123], v[140:143], v[148:151], v[120:123]
	v_mfma_f32_16x16x32_bf16 v[116:119], v[144:147], v[148:151], v[116:119]
	v_mfma_f32_16x16x32_bf16 v[112:115], v[152:155], v[148:151], v[112:115]
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_16x16x32_bf16 v[104:107], v[136:139], v[156:159], v[104:107]
	v_mfma_f32_16x16x32_bf16 v[100:103], v[140:143], v[156:159], v[100:103]
	v_mfma_f32_16x16x32_bf16 v[96:99], v[144:147], v[156:159], v[96:99]
	v_mfma_f32_16x16x32_bf16 v[92:95], v[152:155], v[156:159], v[92:95]
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_16x16x32_bf16 v[88:91], v[136:139], v[160:163], v[88:91]
	v_mfma_f32_16x16x32_bf16 v[84:87], v[140:143], v[160:163], v[84:87]
	v_mfma_f32_16x16x32_bf16 v[80:83], v[144:147], v[160:163], v[80:83]
	v_mfma_f32_16x16x32_bf16 v[76:79], v[152:155], v[160:163], v[76:79]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x32_bf16 v[72:75], v[136:139], v[164:167], v[72:75]
	v_mfma_f32_16x16x32_bf16 v[68:71], v[140:143], v[164:167], v[68:71]
	v_mfma_f32_16x16x32_bf16 v[64:67], v[144:147], v[164:167], v[64:67]
	v_mfma_f32_16x16x32_bf16 v[60:63], v[152:155], v[164:167], v[60:63]

	; load A4-A7, reuse A0-A3 registers
	ds_read_b128 v[148:151], v134 offset:8192
	ds_read_b128 v[156:159], v134 offset:10240
	ds_read_b128 v[160:163], v134 offset:12288
	ds_read_b128 v[164:167], v134 offset:14336

	; wait A4-A7, compute [A4, A5, A6, A7] x [B0, B1, B2, B3]
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_16x16x32_bf16 v[56:59], v[136:139], v[148:151], v[56:59]
	v_mfma_f32_16x16x32_bf16 v[52:55], v[140:143], v[148:151], v[52:55]
	v_mfma_f32_16x16x32_bf16 v[48:51], v[144:147], v[148:151], v[48:51]
	v_mfma_f32_16x16x32_bf16 v[44:47], v[152:155], v[148:151], v[44:47]
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_16x16x32_bf16 v[40:43], v[136:139], v[156:159], v[40:43]
	v_mfma_f32_16x16x32_bf16 v[36:39], v[140:143], v[156:159], v[36:39]
	v_mfma_f32_16x16x32_bf16 v[32:35], v[144:147], v[156:159], v[32:35]
	v_mfma_f32_16x16x32_bf16 v[28:31], v[152:155], v[156:159], v[28:31]
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_16x16x32_bf16 v[24:27], v[136:139], v[160:163], v[24:27]
	v_mfma_f32_16x16x32_bf16 v[20:23], v[140:143], v[160:163], v[20:23]
	v_mfma_f32_16x16x32_bf16 v[16:19], v[144:147], v[160:163], v[16:19]
	v_mfma_f32_16x16x32_bf16 v[12:15], v[152:155], v[160:163], v[12:15]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x32_bf16 v[8:11], v[136:139], v[164:167], v[8:11]
	v_mfma_f32_16x16x32_bf16 v[4:7], v[140:143], v[164:167], v[4:7]
	v_mfma_f32_16x16x32_bf16 v[0:3], v[144:147], v[164:167], v[0:3]
	v_mfma_f32_16x16x32_bf16 v[124:127], v[152:155], v[164:167], v[124:127]

	s_cbranch_scc1 .LBB0_1
	s_load_dwordx2 s[2:3], s[0:1], 0x30
	s_nop 0
	s_load_dwordx2 s[0:1], s[0:1], 0x40
	s_add_i32 s13, s13, s14
	v_or_b32_e32 v129, s13, v130
	s_lshl_b32 s4, s12, 6
	v_lshrrev_b32_e32 v130, 2, v128
	v_and_or_b32 v130, v130, 12, s4
	s_ashr_i32 s4, s13, 31
	v_or_b32_e32 v130, s15, v130
	s_waitcnt lgkmcnt(0)
	s_mul_i32 s4, s0, s4
	v_mul_lo_u32 v131, s1, v129
	v_mad_u64_u32 v[132:133], s[6:7], s0, v129, 0
	v_add3_u32 v133, v133, s4, v131
	v_ashrrev_i32_e32 v131, 31, v130
	v_lshl_add_u64 v[132:133], v[132:133], 1, s[2:3]
	v_lshlrev_b64 v[130:131], 1, v[130:131]
	v_lshl_add_u64 v[132:133], v[132:133], 0, v[130:131]
	v_cvt_pk_bf16_f32 v111, v110, v111
	v_cvt_pk_bf16_f32 v110, v108, v109
	v_cvt_pk_bf16_f32 v109, v122, v123
	v_cvt_pk_bf16_f32 v108, v120, v121
	global_store_dwordx2 v[132:133], v[108:109], off offset:32
	v_cvt_pk_bf16_f32 v109, v118, v119
	v_cvt_pk_bf16_f32 v108, v116, v117
	global_store_dwordx2 v[132:133], v[108:109], off offset:64
	v_cvt_pk_bf16_f32 v109, v114, v115
	v_cvt_pk_bf16_f32 v108, v112, v113
	global_store_dwordx2 v[132:133], v[108:109], off offset:96
	v_or_b32_e32 v108, 16, v129
	global_store_dwordx2 v[132:133], v[110:111], off
	v_mul_lo_u32 v110, s1, v108
	v_mad_u64_u32 v[108:109], s[6:7], s0, v108, 0
	v_add3_u32 v109, v109, s4, v110
	v_lshl_add_u64 v[108:109], v[108:109], 1, s[2:3]
	v_lshl_add_u64 v[108:109], v[108:109], 0, v[130:131]
	v_cvt_pk_bf16_f32 v95, v94, v95
	v_cvt_pk_bf16_f32 v94, v92, v93
	v_or_b32_e32 v92, 32, v129
	global_store_dwordx2 v[108:109], v[94:95], off offset:96
	v_mul_lo_u32 v94, s1, v92
	v_mad_u64_u32 v[92:93], s[6:7], s0, v92, 0
	v_add3_u32 v93, v93, s4, v94
	v_lshl_add_u64 v[92:93], v[92:93], 1, s[2:3]
	v_cvt_pk_bf16_f32 v107, v106, v107
	v_cvt_pk_bf16_f32 v106, v104, v105
	v_cvt_pk_bf16_f32 v103, v102, v103
	v_cvt_pk_bf16_f32 v102, v100, v101
	v_cvt_pk_bf16_f32 v99, v98, v99
	v_cvt_pk_bf16_f32 v98, v96, v97
	v_lshl_add_u64 v[92:93], v[92:93], 0, v[130:131]
	v_cvt_pk_bf16_f32 v79, v78, v79
	v_cvt_pk_bf16_f32 v78, v76, v77
	global_store_dwordx2 v[108:109], v[106:107], off
	global_store_dwordx2 v[108:109], v[102:103], off offset:32
	global_store_dwordx2 v[108:109], v[98:99], off offset:64
	global_store_dwordx2 v[92:93], v[78:79], off offset:96
	v_or_b32_e32 v78, s13, v128
	v_or_b32_e32 v76, 48, v78
	v_mul_lo_u32 v79, s1, v76
	v_mad_u64_u32 v[76:77], s[6:7], s0, v76, 0
	v_add3_u32 v77, v77, s4, v79
	v_lshl_add_u64 v[76:77], v[76:77], 1, s[2:3]
	v_cvt_pk_bf16_f32 v91, v90, v91
	v_cvt_pk_bf16_f32 v90, v88, v89
	v_cvt_pk_bf16_f32 v87, v86, v87
	v_cvt_pk_bf16_f32 v86, v84, v85
	v_cvt_pk_bf16_f32 v83, v82, v83
	v_cvt_pk_bf16_f32 v82, v80, v81
	v_lshl_add_u64 v[76:77], v[76:77], 0, v[130:131]
	v_cvt_pk_bf16_f32 v63, v62, v63
	v_cvt_pk_bf16_f32 v62, v60, v61
	v_or_b32_e32 v60, 64, v129
	global_store_dwordx2 v[92:93], v[90:91], off
	global_store_dwordx2 v[92:93], v[86:87], off offset:32
	global_store_dwordx2 v[92:93], v[82:83], off offset:64
	global_store_dwordx2 v[76:77], v[62:63], off offset:96
	v_mul_lo_u32 v62, s1, v60
	v_mad_u64_u32 v[60:61], s[6:7], s0, v60, 0
	v_add3_u32 v61, v61, s4, v62
	v_lshl_add_u64 v[60:61], v[60:61], 1, s[2:3]
	v_cvt_pk_bf16_f32 v75, v74, v75
	v_cvt_pk_bf16_f32 v74, v72, v73
	v_cvt_pk_bf16_f32 v71, v70, v71
	v_cvt_pk_bf16_f32 v70, v68, v69
	v_cvt_pk_bf16_f32 v67, v66, v67
	v_cvt_pk_bf16_f32 v66, v64, v65
	v_lshl_add_u64 v[60:61], v[60:61], 0, v[130:131]
	v_cvt_pk_bf16_f32 v47, v46, v47
	v_cvt_pk_bf16_f32 v46, v44, v45
	v_or_b32_e32 v44, 0x50, v129
	global_store_dwordx2 v[76:77], v[74:75], off
	global_store_dwordx2 v[76:77], v[70:71], off offset:32
	global_store_dwordx2 v[76:77], v[66:67], off offset:64
	global_store_dwordx2 v[60:61], v[46:47], off offset:96
	v_mul_lo_u32 v46, s1, v44
	v_mad_u64_u32 v[44:45], s[6:7], s0, v44, 0
	v_add3_u32 v45, v45, s4, v46
	v_lshl_add_u64 v[44:45], v[44:45], 1, s[2:3]
	v_cvt_pk_bf16_f32 v59, v58, v59
	v_cvt_pk_bf16_f32 v58, v56, v57
	v_cvt_pk_bf16_f32 v55, v54, v55
	v_cvt_pk_bf16_f32 v54, v52, v53
	v_cvt_pk_bf16_f32 v51, v50, v51
	v_cvt_pk_bf16_f32 v50, v48, v49
	v_lshl_add_u64 v[44:45], v[44:45], 0, v[130:131]
	v_cvt_pk_bf16_f32 v31, v30, v31
	v_cvt_pk_bf16_f32 v30, v28, v29
	v_or_b32_e32 v28, 0x60, v129
	global_store_dwordx2 v[60:61], v[58:59], off
	global_store_dwordx2 v[60:61], v[54:55], off offset:32
	global_store_dwordx2 v[60:61], v[50:51], off offset:64
	global_store_dwordx2 v[44:45], v[30:31], off offset:96
	v_mul_lo_u32 v30, s1, v28
	v_mad_u64_u32 v[28:29], s[6:7], s0, v28, 0
	v_add3_u32 v29, v29, s4, v30
	v_lshl_add_u64 v[28:29], v[28:29], 1, s[2:3]
	v_cvt_pk_bf16_f32 v43, v42, v43
	v_cvt_pk_bf16_f32 v42, v40, v41
	v_cvt_pk_bf16_f32 v39, v38, v39
	v_cvt_pk_bf16_f32 v38, v36, v37
	v_cvt_pk_bf16_f32 v35, v34, v35
	v_cvt_pk_bf16_f32 v34, v32, v33
	v_lshl_add_u64 v[28:29], v[28:29], 0, v[130:131]
	v_cvt_pk_bf16_f32 v15, v14, v15
	v_cvt_pk_bf16_f32 v14, v12, v13
	v_or_b32_e32 v12, 0x70, v78
	global_store_dwordx2 v[44:45], v[42:43], off
	global_store_dwordx2 v[44:45], v[38:39], off offset:32
	global_store_dwordx2 v[44:45], v[34:35], off offset:64
	global_store_dwordx2 v[28:29], v[14:15], off offset:96
	v_mul_lo_u32 v14, s1, v12
	v_mad_u64_u32 v[12:13], s[0:1], s0, v12, 0
	v_add3_u32 v13, v13, s4, v14
	v_lshl_add_u64 v[12:13], v[12:13], 1, s[2:3]
	v_cvt_pk_bf16_f32 v27, v26, v27
	v_cvt_pk_bf16_f32 v26, v24, v25
	v_cvt_pk_bf16_f32 v23, v22, v23
	v_cvt_pk_bf16_f32 v22, v20, v21
	v_cvt_pk_bf16_f32 v19, v18, v19
	v_cvt_pk_bf16_f32 v18, v16, v17
	v_lshl_add_u64 v[12:13], v[12:13], 0, v[130:131]
	v_cvt_pk_bf16_f32 v11, v10, v11
	v_cvt_pk_bf16_f32 v10, v8, v9
	v_cvt_pk_bf16_f32 v7, v6, v7
	v_cvt_pk_bf16_f32 v6, v4, v5
	v_cvt_pk_bf16_f32 v3, v2, v3
	v_cvt_pk_bf16_f32 v2, v0, v1
	v_cvt_pk_bf16_f32 v1, v126, v127
	v_cvt_pk_bf16_f32 v0, v124, v125
	global_store_dwordx2 v[28:29], v[26:27], off
	global_store_dwordx2 v[28:29], v[22:23], off offset:32
	global_store_dwordx2 v[28:29], v[18:19], off offset:64
	global_store_dwordx2 v[12:13], v[10:11], off
	global_store_dwordx2 v[12:13], v[6:7], off offset:32
	global_store_dwordx2 v[12:13], v[2:3], off offset:64
	global_store_dwordx2 v[12:13], v[0:1], off offset:96
	s_endpgm
.Lfunc_end0:
	.size	matmul_v2b, .Lfunc_end0-matmul_v2b
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel matmul_v2b
		.amdhsa_group_segment_fixed_size 49152
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 80
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_kernarg_preload_length 0
		.amdhsa_user_sgpr_kernarg_preload_offset 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_enable_private_segment 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 1
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 168
		.amdhsa_next_free_sgpr 96
		.amdhsa_accum_offset 168
		.amdhsa_reserve_vcc 0
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_dx10_clamp 1
		.amdhsa_ieee_mode 1
		.amdhsa_fp16_overflow 0
		.amdhsa_tg_split 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.text

	.set .Lmatmul_v2b.num_vgpr, 168
	.set .Lmatmul_v2b.num_agpr, 0
	.set .Lmatmul_v2b.numbered_sgpr, 27
	.set .Lmatmul_v2b.num_named_barrier, 0
	.set .Lmatmul_v2b.private_seg_size, 0
	.set .Lmatmul_v2b.uses_vcc, 0
	.set .Lmatmul_v2b.uses_flat_scratch, 0
	.set .Lmatmul_v2b.has_dyn_sized_stack, 0
	.set .Lmatmul_v2b.has_recursion, 0
	.set .Lmatmul_v2b.has_indirect_call, 0
	.p2alignl 6, 3212836864
	.fill 256, 4, 3212836864
	.section	.AMDGPU.gpr_maximums,"",@progbits
	.set amdgpu.max_num_vgpr, 0
	.set amdgpu.max_num_agpr, 0
	.set amdgpu.max_num_sgpr, 0
	.set amdgpu.max_num_named_barrier, 0
	.text
	.section	".note.GNU-stack","",@progbits
	.amdgpu_metadata
---
amdhsa.kernels:
  - .agpr_count:     0
    .args:
      - .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .offset:         8
        .size:           16
        .value_kind:     by_value
      - .address_space:  global
        .offset:         24
        .size:           8
        .value_kind:     global_buffer
      - .offset:         32
        .size:           16
        .value_kind:     by_value
      - .address_space:  global
        .offset:         48
        .size:           8
        .value_kind:     global_buffer
      - .offset:         56
        .size:           16
        .value_kind:     by_value
      - .offset:         72
        .size:           4
        .value_kind:     by_value
      - .offset:         76
        .size:           4
        .value_kind:     by_value
    .group_segment_fixed_size: 49152
    .kernarg_segment_align: 8
    .kernarg_segment_size: 80
    .max_flat_workgroup_size: 256
    .name:           matmul_v2b
    .private_segment_fixed_size: 0
    .reqd_workgroup_size:
      - 256
      - 1
      - 1
    .sgpr_count:     33
    .sgpr_spill_count: 0
    .symbol:         matmul_v2b.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     168
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa-unknown-gfx950
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
