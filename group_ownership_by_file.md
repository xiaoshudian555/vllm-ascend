# vllm-ascend 代码归属拆解（按文件维度）

## 各组业务范围

| 组 | 业务范围 | 侧重点 |
|----|---------|--------|
| **vllm框架1组** | CPU侧优化、prefixCache+ChunkPrefill、PD分离、池化、Attn Backend、ACL Graph | 框架侧 |
| **vllm框架2组** | 长序列、投机解码、EPLB、KV Cache | 框架侧 |
| **vllm框架3组** | 调度、FlashComm、RL场景 | 框架侧 |
| **vllm模型1组** | 模型侧并行(DP/TP/EP)、量化、Function Call | 模型侧 |
| **vllm模型2组** | 模型1组全部 + PP并行、Multi-LoRA | 模型侧 |
| **边端推理组** | 310P 平台专属代码 | 全平台 |

---

## vllm框架1组

### vllm_ascend/ 根目录

| 文件 | 说明 |
|------|------|
| `vllm_ascend/ascend_forward_context.py` | 前向传播上下文（chunk prefill 逻辑） |
| `vllm_ascend/batch_invariant.py` | chunk prefill 批量不变性分析 |
| `vllm_ascend/cpu_binding.py` | CPU 亲和性绑定 |
| `vllm_ascend/profiling_config.py` | profiling chunk 配置 |

### vllm_ascend/attention/ (除 kvcomp_attn/)

| 文件 | 说明 |
|------|------|
| `vllm_ascend/attention/__init__.py` | attention backend 入口 |
| `vllm_ascend/attention/attention_v1.py` | v1 attention 实现（GQA/MHA） |
| `vllm_ascend/attention/attention_mask.py` | attention mask 构建 |
| `vllm_ascend/attention/mla_v1.py` | MLA 多头隐式 attention 实现 |
| `vllm_ascend/attention/sfa_v1.py` | Sparse Flash Attention 实现 |
| `vllm_ascend/attention/utils.py` | attention 工具函数 |
| `vllm_ascend/attention/context_parallel/__init__.py` | context parallel 入口 |
| `vllm_ascend/attention/context_parallel/attention_cp.py` | context parallel attention 实现 |
| `vllm_ascend/attention/context_parallel/common_cp.py` | context parallel 通用逻辑 |
| `vllm_ascend/attention/context_parallel/mla_cp.py` | MLA + context parallel |
| `vllm_ascend/attention/context_parallel/sfa_cp.py` | SFA + context parallel |

### vllm_ascend/compilation/

| 文件 | 说明 |
|------|------|
| `vllm_ascend/compilation/__init__.py` | 编译模块入口 |
| `vllm_ascend/compilation/acl_graph.py` | ACL graph 编译捕获 |
| `vllm_ascend/compilation/compiler_interface.py` | 编译器接口 |
| `vllm_ascend/compilation/graph_fusion_pass_manager.py` | graph fusion pass 管理器 |
| `vllm_ascend/compilation/passes/__init__.py` | passes 入口 |
| `vllm_ascend/compilation/passes/allgather_chunk_noop_pass.py` | allgather chunk noop 消除 pass |
| `vllm_ascend/compilation/passes/allreduce_rmsnorm_fusion_pass.py` | allreduce+rmsnorm 融合 pass（vllm模型1组 共担） |
| `vllm_ascend/compilation/passes/base_pattern.py` | pass 基类模式 |
| `vllm_ascend/compilation/passes/noop_elimination.py` | noop 消除 pass |
| `vllm_ascend/compilation/passes/qknorm_rope_fusion_pass.py` | qk norm+rope 融合 pass |

### vllm_ascend/core/

| 文件 | 说明 |
|------|------|
| `vllm_ascend/core/__init__.py` | 调度核心入口 |
| `vllm_ascend/core/profiling_chunk_predictor.py` | profiling chunk 预测器 |
| `vllm_ascend/core/recompute_scheduler.py` | recompute 调度器（vllm框架3组 共担） |
| `vllm_ascend/core/scheduler_dynamic_batch.py` | 动态 batch 调度（vllm框架3组 共担） |
| `vllm_ascend/core/scheduler_profiling_chunk.py` | profiling chunk 调度 |

### vllm_ascend/device_allocator/

| 文件 | 说明 |
|------|------|
| `vllm_ascend/device_allocator/__init__.py` | 设备内存入口 |
| `vllm_ascend/device_allocator/camem.py` | 设备内存分配器（池化） |

### vllm_ascend/distributed/kv_transfer/

| 文件 | 说明 |
|------|------|
| `vllm_ascend/distributed/kv_transfer/__init__.py` | KV 传输入口 |
| `vllm_ascend/distributed/kv_transfer/ascend_multi_connector.py` | ascend 多连接器（PD 分离） |
| `vllm_ascend/distributed/kv_transfer/kv_p2p/__init__.py` | KV P2P 入口 |
| `vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_connector.py` | mooncake KV P2P 连接器 |
| `vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_layerwise_connector.py` | mooncake 逐层 KV P2P |
| `vllm_ascend/distributed/kv_transfer/kv_pool/__init__.py` | KV pool 入口 |
| `vllm_ascend/distributed/kv_transfer/kv_pool/lmcache_ascend_connector.py` | lmcache ascend KV pool 连接器 |
| `vllm_ascend/distributed/kv_transfer/kv_pool/ucm_connector.py` | UCM KV pool 连接器 |
| `vllm_ascend/distributed/kv_transfer/utils/__init__.py` | KV 传输工具入口 |
| `vllm_ascend/distributed/kv_transfer/utils/mooncake_transfer_engine.py` | mooncake 传输引擎 |
| `vllm_ascend/distributed/kv_transfer/utils/utils.py` | KV 传输工具函数 |

### vllm_ascend/ops/（vllm框架1组 部分）

| 文件 | 说明 |
|------|------|
| `vllm_ascend/ops/mla.py` | MLA 算子 |
| `vllm_ascend/ops/weight_prefetch.py` | 权重预取（CPU→NPU 异步搬运） |
| `vllm_ascend/ops/gdn.py` | GDN 门控增量网络算子（vllm框架2组 共担） |
| `vllm_ascend/ops/activation.py` | 激活函数算子（vllm模型1组 共担） |
| `vllm_ascend/ops/layernorm.py` | layernorm/rmsnorm 算子（vllm模型1组 共担） |
| `vllm_ascend/ops/rotary_embedding.py` | RoPE 旋转位置编码（vllm模型1组 共担） |
| `vllm_ascend/ops/triton/batch_memcpy.py` | 批量 memcpy（triton） |
| `vllm_ascend/ops/triton/bincount.py` | bincount 计数（vllm框架2组 共担） |
| `vllm_ascend/ops/triton/layernorm_gated.py` | gated layernorm（vllm模型1组 共担） |
| `vllm_ascend/ops/triton/rope.py` | RoPE（vllm模型1组 共担） |
| `vllm_ascend/ops/triton/batch_invariant/__init__.py` | batch invariant 入口 |
| `vllm_ascend/ops/triton/batch_invariant/matmul.py` | 批量不变 matmul（chunk prefill） |
| `vllm_ascend/ops/triton/batch_invariant/mean.py` | 批量不变 mean |
| `vllm_ascend/ops/triton/batch_invariant/rmsnorm.py` | 批量不变 rmsnorm |
| `vllm_ascend/ops/triton/batch_invariant/softmax.py` | 批量不变 softmax |

### vllm_ascend/sample/

| 文件 | 说明 |
|------|------|
| `vllm_ascend/sample/__init__.py` | 采样模块入口 |
| `vllm_ascend/sample/penalties.py` | 采样惩罚项（frequency/repetition） |
| `vllm_ascend/sample/rejection_sampler.py` | rejection sampler（投机解码采样验证） |
| `vllm_ascend/sample/sampler.py` | 采样器主逻辑（topk/topp） |

### vllm_ascend/worker/（vllm框架1组 部分）

| 文件 | 说明 |
|------|------|
| `vllm_ascend/worker/block_table.py` | block table 管理（KV cache 块映射） |
| `vllm_ascend/worker/model_runner_v1.py` | v1 model runner（模型执行流程控制） |
| `vllm_ascend/worker/npu_input_batch.py` | NPU 输入 batch 准备（prefill/decode） |
| `vllm_ascend/worker/pcp_utils.py` | PCP prefix cache prefill 工具 |
| `vllm_ascend/worker/v2/__init__.py` | v2 worker 入口 |
| `vllm_ascend/worker/v2/aclgraph_utils.py` | v2 ACL graph 工具 |
| `vllm_ascend/worker/v2/attn_utils.py` | v2 attention 工具 |
| `vllm_ascend/worker/v2/block_table.py` | v2 block table |
| `vllm_ascend/worker/v2/input_batch.py` | v2 输入 batch |
| `vllm_ascend/worker/v2/model_runner.py` | v2 model runner（核心 attn+graph 调度） |
| `vllm_ascend/worker/v2/states.py` | v2 状态管理 |
| `vllm_ascend/worker/v2/utils.py` | v2 工具函数 |
| `vllm_ascend/worker/v2/sample/__init__.py` | v2 采样入口 |
| `vllm_ascend/worker/v2/sample/bad_words.py` | v2 bad words 过滤 |
| `vllm_ascend/worker/v2/sample/gumbel.py` | v2 gumbel 采样 |
| `vllm_ascend/worker/v2/sample/logprob.py` | v2 logprob 计算 |
| `vllm_ascend/worker/v2/sample/min_p.py` | v2 min_p 采样 |
| `vllm_ascend/worker/v2/sample/penalties.py` | v2 采样惩罚项 |

### vllm_ascend/patch/（vllm框架1组 相关）

| 文件 | 说明 |
|------|------|
| `vllm_ascend/patch/platform/patch_profiling_chunk.py` | profiling chunk patch |
| `vllm_ascend/patch/worker/patch_cudagraph.py` | cudagraph patch（ACL graph 相关） |
| `vllm_ascend/patch/worker/patch_npugraph_ex_triton.py` | npu graph + triton 扩展 patch |

### csrc/（vllm框架1组）

| 文件/目录 | 说明 |
|------|------|
| `csrc/aclnn_torch_adapter/` | ACL 与 PyTorch 适配层 |
| `csrc/add_rms_norm_bias/` | attention 残差+rmsnorm+bias 融合算子 |
| `csrc/apply_top_k_top_p_custom/` | 采样 topk/topp 自定义算子 |
| `csrc/batch_matmul_transpose/` | attention 批量 matmul 转置算子 |
| `csrc/mla_preprocess/` | MLA attention 预处理算子 |
| `csrc/sparse_flash_attention/` | sparse flash attention 自定义算子 |

---

## vllm框架2组

### vllm_ascend/

| 文件 | 说明 |
|------|------|
| `vllm_ascend/attention/kvcomp_attn/__init__.py` | KV 压缩 attention 入口 |
| `vllm_ascend/attention/kvcomp_attn/attention_utils.py` | KV 压缩 attention 工具 |

### vllm_ascend/eplb/

| 文件 | 说明 |
|------|------|
| `vllm_ascend/eplb/__init__.py` | EPLB 入口 |
| `vllm_ascend/eplb/eplb_updator.py` | EPLB 更新器 |
| `vllm_ascend/eplb/utils.py` | EPLB 工具函数 |
| `vllm_ascend/eplb/adaptor/__init__.py` | EPLB 适配器入口 |
| `vllm_ascend/eplb/adaptor/vllm_adaptor.py` | EPLB vllm 适配器 |
| `vllm_ascend/eplb/core/__init__.py` | EPLB 核心入口 |
| `vllm_ascend/eplb/core/eplb_device_transfer_loader.py` | EPLB 设备传输加载 |
| `vllm_ascend/eplb/core/eplb_utils.py` | EPLB 核心工具 |
| `vllm_ascend/eplb/core/eplb_worker.py` | EPLB worker |
| `vllm_ascend/eplb/core/policy/__init__.py` | EPLB 策略入口 |
| `vllm_ascend/eplb/core/policy/policy_abstract.py` | 负载均衡策略抽象基类 |
| `vllm_ascend/eplb/core/policy/policy_default_eplb.py` | 默认 EPLB 策略 |
| `vllm_ascend/eplb/core/policy/policy_factory.py` | 策略工厂 |
| `vllm_ascend/eplb/core/policy/policy_flashlb.py` | flashlb 策略 |
| `vllm_ascend/eplb/core/policy/policy_random.py` | 随机策略 |
| `vllm_ascend/eplb/core/policy/policy_swift_balancer.py` | swift balancer 策略 |

### vllm_ascend/kv_offload/

| 文件 | 说明 |
|------|------|
| `vllm_ascend/kv_offload/__init__.py` | KV 卸载入口 |
| `vllm_ascend/kv_offload/cpu_npu.py` | KV cache CPU↔NPU 卸载 |
| `vllm_ascend/kv_offload/npu.py` | KV cache NPU 侧管理 |

### vllm_ascend/spec_decode/

| 文件 | 说明 |
|------|------|
| `vllm_ascend/spec_decode/__init__.py` | 投机解码入口 |
| `vllm_ascend/spec_decode/dflash_proposer.py` | dflash proposer |
| `vllm_ascend/spec_decode/draft_proposer.py` | draft proposer 基类 |
| `vllm_ascend/spec_decode/eagle_proposer.py` | eagle proposer |
| `vllm_ascend/spec_decode/medusa_proposer.py` | medusa proposer |
| `vllm_ascend/spec_decode/ngram_proposer.py` | ngram proposer |
| `vllm_ascend/spec_decode/suffix_proposer.py` | suffix proposer |
| `vllm_ascend/spec_decode/utils.py` | 投机解码工具函数 |

### vllm_ascend/ops/（vllm框架2组 部分）

| 文件 | 说明 |
|------|------|
| `vllm_ascend/ops/conv.py` | 卷积算子（长序列 causal conv） |
| `vllm_ascend/ops/gdn.py` | GDN 门控增量网络算子（vllm框架1组 共担） |
| `vllm_ascend/ops/triton/bincount.py` | bincount 计数（vllm框架1组 共担） |
| `vllm_ascend/ops/triton/fused_gdn_gating.py` | GDN fused gating（triton） |
| `vllm_ascend/ops/triton/gdn_chunk_meta.py` | GDN chunk meta 计算（triton） |
| `vllm_ascend/ops/triton/penalty.py` | 采样惩罚项（triton） |
| `vllm_ascend/ops/triton/reject_sample.py` | 投机解码 reject sample（triton） |
| `vllm_ascend/ops/triton/fla/__init__.py` | FLA 算子入口 |
| `vllm_ascend/ops/triton/fla/chunk.py` | FLA chunk 主逻辑 |
| `vllm_ascend/ops/triton/fla/chunk_delta_h.py` | FLA delta_h 计算 |
| `vllm_ascend/ops/triton/fla/chunk_delta_hupdate.py` | FLA delta_h 更新 |
| `vllm_ascend/ops/triton/fla/chunk_o.py` | FLA chunk output |
| `vllm_ascend/ops/triton/fla/chunk_o_update.py` | FLA chunk o 更新 |
| `vllm_ascend/ops/triton/fla/chunk_scaled_dot_kkt.py` | FLA chunk scaled dot KKT |
| `vllm_ascend/ops/triton/fla/cumsum.py` | FLA cumsum |
| `vllm_ascend/ops/triton/fla/fused_qkvzba_split_reshape.py` | FLA fused QKV 拆分重塑 |
| `vllm_ascend/ops/triton/fla/l2norm.py` | FLA L2 norm |
| `vllm_ascend/ops/triton/fla/layernorm_guard.py` | FLA layernorm guard |
| `vllm_ascend/ops/triton/fla/sigmoid_gating.py` | FLA sigmoid gating |
| `vllm_ascend/ops/triton/fla/solve_tril.py` | FLA solve tril |
| `vllm_ascend/ops/triton/fla/utils.py` | FLA 工具函数 |
| `vllm_ascend/ops/triton/fla/wy_fast.py` | FLA wy fast 算法 |
| `vllm_ascend/ops/triton/mamba/__init__.py` | mamba 算子入口 |
| `vllm_ascend/ops/triton/mamba/causal_conv1d.py` | causal conv1d（长序列） |
| `vllm_ascend/ops/triton/mamba/lightning_attn.py` | lightning attention（长序列） |
| `vllm_ascend/ops/triton/mamba/linear_attn.py` | linear attention（长序列） |
| `vllm_ascend/ops/triton/spec_decode/__init__.py` | 投机解码 triton 入口 |
| `vllm_ascend/ops/triton/spec_decode/utils.py` | 投机解码 triton 工具 |

### vllm_ascend/worker/（vllm框架2组 部分）

| 文件 | 说明 |
|------|------|
| `vllm_ascend/worker/kvcomp_utils.py` | KV 压缩工具 |
| `vllm_ascend/worker/v2/spec_decode/eagle/__init__.py` | v2 eagle 投机入口 |
| `vllm_ascend/worker/v2/spec_decode/eagle/aclgraph.py` | v2 eagle ACL graph |
| `vllm_ascend/worker/v2/spec_decode/eagle/speculator.py` | v2 eagle speculator |

### vllm_ascend/patch/（vllm框架2组 相关）

| 文件 | 说明 |
|------|------|
| `vllm_ascend/patch/platform/patch_kv_cache_interface.py` | KV cache 接口 patch |
| `vllm_ascend/patch/platform/patch_mamba_config.py` | mamba 模型配置 patch |
| `vllm_ascend/patch/platform/patch_minimax_m2_config.py` | minimax m2 配置 patch |
| `vllm_ascend/patch/worker/patch_bert.py` | bert 模型 patch |
| `vllm_ascend/patch/worker/patch_deepseek_mtp.py` | deepseek mtp patch（投机解码） |
| `vllm_ascend/patch/worker/patch_draft_quarot.py` | draft quarot patch（投机量化） |
| `vllm_ascend/patch/worker/patch_gdn_attn.py` | GDN attention patch |
| `vllm_ascend/patch/worker/patch_mamba_utils.py` | mamba 工具 patch |
| `vllm_ascend/patch/worker/patch_minimax_m2.py` | minimax m2 模型 patch |
| `vllm_ascend/patch/worker/patch_minimax_m2_linear_attn.py` | minimax m2 linear attention patch |
| `vllm_ascend/patch/worker/patch_qwen3_next_mtp.py` | qwen3 next mtp patch（投机） |
| `vllm_ascend/patch/worker/patch_rejection_sampler.py` | rejection sampler patch（投机） |
| `vllm_ascend/patch/worker/patch_routed_experts_capturer.py` | routed experts capturer patch（EPLB） |

### csrc/（vllm框架2组）

| 文件/目录 | 说明 |
|------|------|
| `csrc/causal_conv1d/` | 长序列 causal conv1d（CANN 算子） |
| `csrc/copy_and_expand_eagle_inputs/` | eagle 投机解码输入扩展算子 |
| `csrc/hamming_dist_top_k/` | 投机解码 hamming 距离 topk 算子 |
| `csrc/lightning_indexer_quant/` | 长序列 lightning indexer 量化算子 |
| `csrc/lightning_indexer_vllm/` | 长序列 lightning indexer vllm 算子 |
| `csrc/recurrent_gated_delta_rule/` | 长序列 FLA 递归门控增量规则算子 |
| `csrc/reshape_and_cache_bnsd/` | KV cache reshape & cache 算子 |
| `csrc/transpose_kv_cache_by_block/` | KV cache block 转置算子 |

---

## vllm框架3组

### vllm_ascend/ 根目录

| 文件 | 说明 |
|------|------|
| `vllm_ascend/flash_common3_context.py` | flashcomm 上下文 |

### vllm_ascend/distributed/device_communicators/

| 文件 | 说明 |
|------|------|
| `vllm_ascend/distributed/device_communicators/__init__.py` | 设备通信入口 |
| `vllm_ascend/distributed/device_communicators/npu_communicator.py` | NPU 通信器（HCCL 封装） |
| `vllm_ascend/distributed/device_communicators/pyhccl.py` | HCCL Python 绑定 |
| `vllm_ascend/distributed/device_communicators/pyhccl_wrapper.py` | HCCL wrapper 工具 |

### vllm_ascend/ops/（vllm框架3组 部分）

| 文件 | 说明 |
|------|------|
| `vllm_ascend/ops/flashcomm2_oshard_manager.py` | flashcomm2 O shard 管理 |
| `vllm_ascend/ops/fused_moe/__init__.py` | fused MoE 入口 |
| `vllm_ascend/ops/fused_moe/comm_utils.py` | MoE 通信工具函数 |
| `vllm_ascend/ops/fused_moe/experts_selector.py` | MoE experts 选择器 |
| `vllm_ascend/ops/fused_moe/fused_moe.py` | fused MoE 前向主逻辑 |
| `vllm_ascend/ops/fused_moe/moe_comm_method.py` | MoE 通信方式（flashcomm/TP） |
| `vllm_ascend/ops/fused_moe/moe_mlp.py` | MoE MLP 层 |
| `vllm_ascend/ops/fused_moe/moe_runtime_args.py` | MoE 运行时参数 |
| `vllm_ascend/ops/fused_moe/moe_stage_contracts.py` | MoE stage 契约 |
| `vllm_ascend/ops/fused_moe/moe_stage_params.py` | MoE stage 参数 |
| `vllm_ascend/ops/fused_moe/prepare_finalize.py` | MoE prepare/finalize |
| `vllm_ascend/ops/fused_moe/token_dispatcher.py` | MoE token 分发与负载调度 |

### vllm_ascend/core/（vllm框架3组 共担）

| 文件 | 说明 |
|------|------|
| `vllm_ascend/core/recompute_scheduler.py` | recompute 调度器（vllm框架1组 共担） |
| `vllm_ascend/core/scheduler_dynamic_batch.py` | 动态 batch 调度（vllm框架1组 共担） |

### vllm_ascend/patch/（vllm框架3组 相关）

| 文件 | 说明 |
|------|------|
| `vllm_ascend/patch/platform/patch_balance_schedule.py` | 负载均衡调度 patch |
| `vllm_ascend/patch/platform/patch_sched_yield.py` | 调度 yield patch |
| `vllm_ascend/patch/worker/_hccl_pg_registry.py` | HCCL process group 注册表 |
| `vllm_ascend/patch/worker/patch_bailing_moe_linear.py` | bailing MoE linear patch（flashcomm） |

### csrc/（vllm框架3组）

| 文件/目录 | 说明 |
|------|------|
| `csrc/dispatch_ffn_combine/` | MoE dispatch+ffn+combine 三合一融合算子 |
| `csrc/dispatch_ffn_combine_bf16/` | MoE 三合一融合算子 bf16 版本 |
| `csrc/dispatch_ffn_combine_w4_a8/` | MoE 三合一融合算子 w4a8 量化版本 |
| `csrc/dispatch_gmm_combine_decode/` | decode 阶段 GMM combine 算子 |
| `csrc/dispatch_layout/` | dispatch 布局转换算子 |
| `csrc/moe_combine_normal/` | MoE combine 算子 |
| `csrc/moe_dispatch_normal/` | MoE dispatch 算子 |
| `csrc/moe_gating_top_k/` | MoE gating top-k 算子 |
| `csrc/moe_grouped_matmul/` | MoE grouped matmul 算子 |
| `csrc/moe_init_routing_custom/` | MoE routing 初始化算子 |
| `csrc/notify_dispatch/` | 调度通知 dispatch 算子 |

---

## vllm模型1组

### vllm_ascend/ 根目录

| 文件 | 说明 |
|------|------|
| `vllm_ascend/meta_registration.py` | 模型元信息注册（vllm模型2组 共担） |

### vllm_ascend/device/

| 文件 | 说明 |
|------|------|
| `vllm_ascend/device/__init__.py` | 设备模块入口 |
| `vllm_ascend/device/device_op.py` | 设备操作封装（vllm框架1组 共担） |
| `vllm_ascend/device/mxfp_compat.py` | MXFP 格式兼容 |

### vllm_ascend/distributed/（vllm模型1组 部分）

| 文件 | 说明 |
|------|------|
| `vllm_ascend/distributed/__init__.py` | 分布式模块入口 |
| `vllm_ascend/distributed/parallel_state.py` | DP/TP/EP 并行状态管理（vllm模型2组 共担） |
| `vllm_ascend/distributed/utils.py` | 分布式工具函数（vllm模型2组 共担） |

### vllm_ascend/ops/（vllm模型1组 部分）

| 文件 | 说明 |
|------|------|
| `vllm_ascend/ops/activation.py` | 激活函数算子（vllm框架1组 共担） |
| `vllm_ascend/ops/layer_shard_linear.py` | DP 层切分线性层 |
| `vllm_ascend/ops/layernorm.py` | layernorm/rmsnorm 算子（vllm框架1组 共担） |
| `vllm_ascend/ops/linear.py` | TP 线性层 |
| `vllm_ascend/ops/linear_op.py` | 线性层底层实现 |
| `vllm_ascend/ops/rotary_embedding.py` | RoPE 旋转位置编码（vllm框架1组 共担） |
| `vllm_ascend/ops/vocab_parallel_embedding.py` | TP 词表并行 embedding |
| `vllm_ascend/ops/triton/layernorm_gated.py` | gated layernorm（vllm框架1组 共担） |
| `vllm_ascend/ops/triton/rope.py` | RoPE（vllm框架1组 共担） |
| `vllm_ascend/ops/triton/activation/__init__.py` | triton 激活入口 |
| `vllm_ascend/ops/triton/activation/swiglu_quant.py` | SwiGLU + 量化融合 |
| `vllm_ascend/ops/triton/linearnorm/__init__.py` | linearnorm 入口 |
| `vllm_ascend/ops/triton/linearnorm/split_qkv_rmsnorm_mrope.py` | QKV split+rmsnorm+mrope 融合 |
| `vllm_ascend/ops/triton/linearnorm/split_qkv_rmsnorm_rope.py` | QKV split+rmsnorm+rope 融合 |
| `vllm_ascend/ops/triton/linearnorm/split_qkv_tp_rmsnorm_rope.py` | QKV split+rmsnorm+rope 融合（TP 变体） |

### vllm_ascend/quantization/

| 文件 | 说明 |
|------|------|
| `vllm_ascend/quantization/__init__.py` | 量化模块入口 |
| `vllm_ascend/quantization/compressed_tensors_config.py` | 压缩张量配置 |
| `vllm_ascend/quantization/method_adapters.py` | 量化方法适配器 |
| `vllm_ascend/quantization/modelslim_config.py` | modelslim 量化配置 |
| `vllm_ascend/quantization/quant_parser.py` | 量化参数解析 |
| `vllm_ascend/quantization/quant_type.py` | 量化类型定义 |
| `vllm_ascend/quantization/utils.py` | 量化工具函数 |
| `vllm_ascend/quantization/methods/__init__.py` | 量化方法入口 |
| `vllm_ascend/quantization/methods/base.py` | 量化方法基类 |
| `vllm_ascend/quantization/methods/kv_c8.py` | KV cache int8 量化 |
| `vllm_ascend/quantization/methods/registry.py` | 量化方法注册表 |
| `vllm_ascend/quantization/methods/w4a16.py` | 权重量化 int4 / 激活 fp16 |
| `vllm_ascend/quantization/methods/w4a4_flatquant.py` | w4a4 flatquant 量化 |
| `vllm_ascend/quantization/methods/w4a4_laos_dynamic.py` | w4a4 laos 动态量化 |
| `vllm_ascend/quantization/methods/w4a4_mxfp4.py` | w4a4 mxfp4 量化 |
| `vllm_ascend/quantization/methods/w4a8.py` | 权重量化 int4 / 激活 int8 |
| `vllm_ascend/quantization/methods/w8a16.py` | 权重量化 int8 / 激活 fp16 |
| `vllm_ascend/quantization/methods/w8a8_dynamic.py` | w8a8 动态量化 |
| `vllm_ascend/quantization/methods/w8a8_mxfp8.py` | w8a8 mxfp8 量化 |
| `vllm_ascend/quantization/methods/w8a8_pdmix.py` | w8a8 pdmix 量化 |
| `vllm_ascend/quantization/methods/w8a8_static.py` | w8a8 静态量化 |

### vllm_ascend/compilation/passes/（vllm模型1组 共担）

| 文件 | 说明 |
|------|------|
| `vllm_ascend/compilation/passes/allreduce_rmsnorm_fusion_pass.py` | allreduce+rmsnorm 融合 pass（vllm框架1组 共担） |
| `vllm_ascend/compilation/passes/norm_quant_fusion_pass.py` | norm+quant 融合 pass |

### vllm_ascend/worker/v2/model_states/（vllm模型1组 共担）

| 文件 | 说明 |
|------|------|
| `vllm_ascend/worker/v2/model_states/__init__.py` | 模型状态入口（vllm模型2组 共担） |
| `vllm_ascend/worker/v2/model_states/default.py` | 默认模型状态管理（vllm模型2组 共担） |

### vllm_ascend/patch/（vllm模型1组 相关）

| 文件 | 说明 |
|------|------|
| `vllm_ascend/patch/platform/patch_distributed.py` | 分布式 patch（vllm模型2组 共担） |
| `vllm_ascend/patch/platform/patch_torch_accelerator.py` | torch 加速器 patch（vllm模型2组 共担） |
| `vllm_ascend/patch/worker/patch_distributed.py` | 分布式 worker patch（vllm模型2组 共担） |
| `vllm_ascend/patch/worker/patch_gqa_c8.py` | GQA C8 patch（量化） |
| `vllm_ascend/patch/worker/patch_huanyuan_vl.py` | huanyuan VL 模型 patch |
| `vllm_ascend/patch/worker/patch_multimodal_merge.py` | 多模态 merge patch |
| `vllm_ascend/patch/worker/patch_qwen3_dflash.py` | qwen3 dflash patch |
| `vllm_ascend/patch/worker/patch_qwen3vl.py` | qwen3vl 模型 patch |
| `vllm_ascend/patch/worker/patch_weight_utils.py` | 权重工具 patch（量化/并行） |

### csrc/（vllm模型1组）

| 文件/目录 | 说明 |
|------|------|
| `csrc/grouped_matmul_swiglu_quant_weight_nz_tensor_list/` | 量化分组 matmul+swiglu 融合算子 |
| `csrc/matmul_allreduce_add_rmsnorm/` | TP matmul+allreduce+rmsnorm 三合一融合算子 |

---

## vllm模型2组

> 包含 vllm模型1组 全部文件，外加以下文件：

### vllm_ascend/lora/

| 文件 | 说明 |
|------|------|
| `vllm_ascend/lora/__init__.py` | LoRA 模块入口 |
| `vllm_ascend/lora/lora_ops.py` | LoRA 算子（sgmv/add） |
| `vllm_ascend/lora/punica_npu.py` | punica NPU Multi-LoRA 实现 |
| `vllm_ascend/lora/utils.py` | LoRA 工具函数 |

### vllm_ascend/model_loader/

| 文件 | 说明 |
|------|------|
| `vllm_ascend/model_loader/__init__.py` | 模型加载入口 |
| `vllm_ascend/model_loader/netloader/__init__.py` | netloader 入口 |
| `vllm_ascend/model_loader/netloader/load.py` | 模型加载入口 |
| `vllm_ascend/model_loader/netloader/netloader.py` | 网络加载器（PP 模型分发） |
| `vllm_ascend/model_loader/netloader/utils.py` | netloader 工具 |
| `vllm_ascend/model_loader/netloader/executor/__init__.py` | netloader 执行器入口 |
| `vllm_ascend/model_loader/netloader/executor/elastic_load.py` | 弹性加载执行器（PP） |
| `vllm_ascend/model_loader/netloader/executor/netloader_pg.py` | netloader process group |
| `vllm_ascend/model_loader/netloader/interaction/__init__.py` | netloader 交互入口 |
| `vllm_ascend/model_loader/netloader/interaction/elastic.py` | 弹性交互协议（PP） |
| `vllm_ascend/model_loader/rfork/__init__.py` | rfork 入口 |
| `vllm_ascend/model_loader/rfork/rfork_loader.py` | rfork 加载器（PP 权重分发） |
| `vllm_ascend/model_loader/rfork/rfork_worker.py` | rfork worker |
| `vllm_ascend/model_loader/rfork/seed_protocol.py` | rfork 种子文件协议 |
| `vllm_ascend/model_loader/rfork/seed_server.py` | rfork 种子文件 server |
| `vllm_ascend/model_loader/rfork/transfer_backend.py` | rfork 传输后端 |

### vllm_ascend/compilation/passes/（vllm模型2组 部分）

| 文件 | 说明 |
|------|------|
| `vllm_ascend/compilation/passes/sequence_parallelism.py` | 序列并行 pass（PP） |
| `vllm_ascend/compilation/passes/sequence_parallelism_moe.py` | 序列并行 MoE pass（vllm框架3组 共担） |

### vllm_ascend/patch/（vllm模型2组 额外）

| 文件 | 说明 |
|------|------|
| `vllm_ascend/patch/worker/patch_kimi_k25.py` | kimi k25 模型 patch |
| `vllm_ascend/patch/worker/patch_module.py` | module patch（PP/LoRA） |
| `vllm_ascend/patch/worker/patch_qwen3_5.py` | qwen3.5 模型 patch |
| `vllm_ascend/patch/worker/patch_triton.py` | triton patch |

---

## 边端推理组 — 310P 平台专属

### vllm_ascend/_310p/

| 文件 | 说明 |
|------|------|
| `vllm_ascend/_310p/__init__.py` | 310P 平台入口 |
| `vllm_ascend/_310p/block_table.py` | 310P block table |
| `vllm_ascend/_310p/model_runner_310p.py` | 310P model runner |
| `vllm_ascend/_310p/npu_input_batch.py` | 310P NPU 输入 batch |
| `vllm_ascend/_310p/sharded_state_loader_310p.py` | 310P 分片状态加载 |
| `vllm_ascend/_310p/worker_310p.py` | 310P worker |
| `vllm_ascend/_310p/attention/__init__.py` | 310P attention 入口 |
| `vllm_ascend/_310p/attention/attention_mask.py` | 310P attention mask |
| `vllm_ascend/_310p/attention/attention_v1.py` | 310P attention 实现 |
| `vllm_ascend/_310p/attention/metadata_builder.py` | 310P 元数据构建 |
| `vllm_ascend/_310p/fused_moe/__init__.py` | 310P fused MoE 入口 |
| `vllm_ascend/_310p/fused_moe/experts_selector.py` | 310P MoE experts 选择器 |
| `vllm_ascend/_310p/fused_moe/fused_moe.py` | 310P fused MoE 实现 |
| `vllm_ascend/_310p/fused_moe/moe_comm_method.py` | 310P MoE 通信方式 |
| `vllm_ascend/_310p/fused_moe/moe_mlp.py` | 310P MoE MLP |
| `vllm_ascend/_310p/fused_moe/token_dispatcher.py` | 310P MoE token 分发 |
| `vllm_ascend/_310p/ops/__init__.py` | 310P 算子入口 |
| `vllm_ascend/_310p/ops/activation.py` | 310P 激活函数算子 |
| `vllm_ascend/_310p/ops/causal_conv1d.py` | 310P causal conv1d |
| `vllm_ascend/_310p/ops/conv.py` | 310P 卷积算子 |
| `vllm_ascend/_310p/ops/layernorm.py` | 310P layernorm 算子 |
| `vllm_ascend/_310p/ops/mm_encoder_attention.py` | 310P 多模态 encoder attention |
| `vllm_ascend/_310p/ops/rotary_embedding.py` | 310P RoPE |
| `vllm_ascend/_310p/ops/vocab_parallel_embedding.py` | 310P TP 词表并行 |
| `vllm_ascend/_310p/ops/fla/__init__.py` | 310P FLA 入口 |
| `vllm_ascend/_310p/ops/fla/chunk_gated_delta_rule.py` | 310P chunk gated delta rule |
| `vllm_ascend/_310p/ops/fla/fused_gdn_gating.py` | 310P fused GDN gating |
| `vllm_ascend/_310p/ops/fla/fused_recurrent_gated_delta_rule.py` | 310P fused recurrent gated delta rule |
| `vllm_ascend/_310p/ops/fla/gdn_310.py` | 310P GDN 实现 |
| `vllm_ascend/_310p/quantization/__init__.py` | 310P 量化入口 |
| `vllm_ascend/_310p/quantization/modelslim_config.py` | 310P modelslim 量化配置 |
| `vllm_ascend/_310p/quantization/methods/__init__.py` | 310P 量化方法入口 |
| `vllm_ascend/_310p/quantization/methods/registry.py` | 310P 量化方法注册 |
| `vllm_ascend/_310p/quantization/methods/w8a8_dynamic.py` | 310P w8a8 动态量化 |
| `vllm_ascend/_310p/quantization/methods/w8a8_static.py` | 310P w8a8 静态量化 |
| `vllm_ascend/_310p/quantization/methods/w8a8s.py` | 310P w8a8s 量化 |
| `vllm_ascend/_310p/quantization/methods/w8a8sc.py` | 310P w8a8sc 量化 |
| `vllm_ascend/_310p/sample/__init__.py` | 310P 采样入口 |
| `vllm_ascend/_310p/sample/sampler.py` | 310P 采样器 |

### vllm_ascend/patch/（边端推理组）

| 文件 | 说明 |
|------|------|
| `vllm_ascend/patch/platform/patch_mamba_config_310.py` | 310P mamba 配置 patch |

### csrc/（边端推理组）

| 文件/目录 | 说明 |
|------|------|
| `csrc/causal_conv1d_v310/` | 310P causal conv1d |
| `csrc/recurrent_gated_delta_rule_v310/` | 310P 递归门控增量规则算子 |

---

## 公共文件（多组共用）

| 文件 | 说明 |
|------|------|
| `vllm_ascend/__init__.py` | 包入口 |
| `vllm_ascend/platform.py` | 平台初始化 |
| `vllm_ascend/ascend_config.py` | ascend 全局配置 |
| `vllm_ascend/envs.py` | 环境变量管理 |
| `vllm_ascend/utils.py` | 通用工具函数 |
| `vllm_ascend/ops/__init__.py` | 算子模块入口 |
| `vllm_ascend/ops/mm_encoder_attention.py` | 多模态 encoder attention |
| `vllm_ascend/ops/qwen2_decoder.py` | qwen2 decoder 专用算子 |
| `vllm_ascend/ops/register_custom_ops.py` | 自定义算子注册 |
| `vllm_ascend/ops/rel_pos_attention.py` | 相对位置 attention |
| `vllm_ascend/ops/triton/__init__.py` | triton 算子入口 |
| `vllm_ascend/ops/triton/muls_add.py` | mul+add 融合（triton） |
| `vllm_ascend/ops/triton/triton_utils.py` | triton 工具函数 |
| `vllm_ascend/patch/__init__.py` | monkey patch 入口 |
| `vllm_ascend/patch/platform/__init__.py` | platform patch 入口 |
| `vllm_ascend/patch/platform/patch_multiproc_executor.py` | 多进程执行器 patch |
| `vllm_ascend/patch/worker/__init__.py` | worker patch 入口 |
| `vllm_ascend/worker/__init__.py` | worker 模块入口 |
| `vllm_ascend/worker/worker.py` | worker 主入口 |
| `vllm_ascend/compilation/passes/muls_add_pass.py` | mul+add 融合 pass |
| `vllm_ascend/compilation/passes/utils/__init__.py` | pass 工具 |
| `vllm_ascend/core/__init__.py` | 调度核心入口 |
| `vllm_ascend/device/__init__.py` | 设备模块入口 |
| `vllm_ascend/distributed/__init__.py` | 分布式模块入口 |
| `vllm_ascend/xlite/__init__.py` | 轻量推理入口 |
| `vllm_ascend/xlite/utils.py` | xlite 工具 |
| `vllm_ascend/xlite/xlite.py` | xlite 主逻辑 |
| `vllm_ascend/xlite/xlite_model_runner.py` | xlite model runner |
| `vllm_ascend/xlite/xlite_worker.py` | xlite worker |
| `csrc/CMakeLists.txt` | 构建入口 |
| `csrc/kernels/` | 通用 kernel 基础库 |
| `csrc/utils/` | 通用工具（inc/, src/） |
| `csrc/cmake/` | 构建系统脚本 |
| `csrc/third_party/` | 第三方依赖（catlass 等） |
