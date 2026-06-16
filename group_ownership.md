# vllm-ascend 代码归属拆解

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

## vllm_ascend/

```
vllm_ascend/
├── __init__.py                           公共              包入口
├── platform.py                           公共              平台初始化
├── ascend_config.py                      公共              ascend 全局配置
├── envs.py                               公共              环境变量管理
├── utils.py                              公共              通用工具函数
├── meta_registration.py                  vllm模型1组 / vllm模型2组   模型元信息注册
├── ascend_forward_context.py             vllm框架1组       前向传播上下文（chunk prefill 逻辑）
├── batch_invariant.py                    vllm框架1组       chunk prefill 批量不变性分析
├── cpu_binding.py                        vllm框架1组       CPU 亲和性绑定
├── flash_common3_context.py              vllm框架3组       flashcomm 上下文
├── profiling_config.py                   vllm框架1组       profiling chunk 配置
│
├── attention/
│   ├── __init__.py                       vllm框架1组       attention backend 入口
│   ├── attention_v1.py                   vllm框架1组       v1 attention 实现（GQA/MHA）
│   ├── attention_mask.py                 vllm框架1组       attention mask 构建
│   ├── mla_v1.py                         vllm框架1组       MLA 多头隐式 attention 实现
│   ├── sfa_v1.py                         vllm框架1组       Sparse Flash Attention 实现
│   ├── utils.py                          vllm框架1组       attention 工具函数
│   ├── context_parallel/
│   │   ├── __init__.py                   vllm框架1组
│   │   ├── attention_cp.py               vllm框架1组       context parallel attention 实现
│   │   ├── common_cp.py                  vllm框架1组       context parallel 通用逻辑
│   │   ├── mla_cp.py                     vllm框架1组       MLA + context parallel
│   │   └── sfa_cp.py                     vllm框架1组       SFA + context parallel
│   └── kvcomp_attn/
│       ├── __init__.py                   vllm框架2组
│       └── attention_utils.py            vllm框架2组       KV 压缩 attention 工具
│
├── compilation/
│   ├── __init__.py                       vllm框架1组
│   ├── acl_graph.py                      vllm框架1组       ACL graph 编译捕获
│   ├── compiler_interface.py             vllm框架1组       编译器接口
│   ├── graph_fusion_pass_manager.py      vllm框架1组       graph fusion pass 管理器
│   └── passes/
│       ├── __init__.py                   vllm框架1组
│       ├── allgather_chunk_noop_pass.py  vllm框架1组       allgather chunk noop 消除 pass
│       ├── allreduce_rmsnorm_fusion_pass.py  vllm模型1组 / vllm框架1组   allreduce+rmsnorm 融合 pass
│       ├── base_pattern.py               vllm框架1组       pass 基类模式
│       ├── muls_add_pass.py              公共              mul+add 融合 pass
│       ├── noop_elimination.py           vllm框架1组       noop 消除 pass
│       ├── norm_quant_fusion_pass.py     vllm模型1组       norm+quant 融合 pass
│       ├── qknorm_rope_fusion_pass.py    vllm框架1组       qk norm+rope 融合 pass
│       ├── sequence_parallelism.py       vllm模型2组       序列并行 pass（PP）
│       ├── sequence_parallelism_moe.py   vllm模型2组 / vllm框架3组   序列并行 MoE pass（PP+MoE）
│       └── utils/
│           └── __init__.py               公共              pass 工具
│
├── core/
│   ├── __init__.py                       公共              调度核心入口
│   ├── profiling_chunk_predictor.py      vllm框架1组       profiling chunk 预测器
│   ├── recompute_scheduler.py            vllm框架1组 / vllm框架3组   recompute 调度器
│   ├── scheduler_dynamic_batch.py        vllm框架1组 / vllm框架3组   动态 batch 调度
│   └── scheduler_profiling_chunk.py      vllm框架1组       profiling chunk 调度
│
├── device/
│   ├── __init__.py                       公共              设备模块入口
│   ├── device_op.py                      vllm模型1组 / vllm框架1组   设备操作封装
│   └── mxfp_compat.py                    vllm模型1组       MXFP 格式兼容
│
├── device_allocator/
│   ├── __init__.py                       vllm框架1组
│   └── camem.py                          vllm框架1组       设备内存分配器（池化）
│
├── distributed/
│   ├── __init__.py                       公共              分布式模块入口
│   ├── parallel_state.py                 vllm模型1组 / vllm模型2组   DP/TP/EP 并行状态管理
│   ├── utils.py                          vllm模型1组 / vllm模型2组   分布式工具函数
│   ├── device_communicators/
│   │   ├── __init__.py                   vllm框架3组
│   │   ├── npu_communicator.py           vllm框架3组       NPU 通信器（HCCL 封装）
│   │   ├── pyhccl.py                     vllm框架3组       HCCL Python 绑定
│   │   └── pyhccl_wrapper.py             vllm框架3组       HCCL wrapper 工具
│   └── kv_transfer/
│       ├── __init__.py                   vllm框架1组
│       ├── ascend_multi_connector.py     vllm框架1组       ascend 多连接器（PD 分离）
│       ├── kv_p2p/
│       │   ├── __init__.py               vllm框架1组
│       │   ├── mooncake_connector.py     vllm框架1组       mooncake KV P2P 连接器
│       │   └── mooncake_layerwise_connector.py  vllm框架1组   mooncake 逐层 KV P2P
│       ├── kv_pool/
│       │   ├── __init__.py               vllm框架1组
│       │   ├── lmcache_ascend_connector.py  vllm框架1组    lmcache ascend KV pool 连接器
│       │   └── ucm_connector.py          vllm框架1组       UCM KV pool 连接器
│       └── utils/
│           ├── __init__.py               vllm框架1组
│           ├── mooncake_transfer_engine.py  vllm框架1组    mooncake 传输引擎
│           └── utils.py                  vllm框架1组       KV 传输工具函数
│
├── eplb/
│   ├── __init__.py                       vllm框架2组
│   ├── eplb_updator.py                   vllm框架2组       EPLB 更新器
│   ├── utils.py                          vllm框架2组       EPLB 工具函数
│   ├── adaptor/
│   │   ├── __init__.py                   vllm框架2组
│   │   └── vllm_adaptor.py               vllm框架2组       EPLB vllm 适配器
│   └── core/
│       ├── __init__.py                   vllm框架2组
│       ├── eplb_device_transfer_loader.py  vllm框架2组     EPLB 设备传输加载
│       ├── eplb_utils.py                 vllm框架2组       EPLB 核心工具
│       ├── eplb_worker.py                vllm框架2组       EPLB worker
│       └── policy/
│           ├── __init__.py               vllm框架2组
│           ├── policy_abstract.py        vllm框架2组       负载均衡策略抽象基类
│           ├── policy_default_eplb.py    vllm框架2组       默认 EPLB 策略
│           ├── policy_factory.py         vllm框架2组       策略工厂
│           ├── policy_flashlb.py         vllm框架2组       flashlb 策略
│           ├── policy_random.py          vllm框架2组       随机策略
│           └── policy_swift_balancer.py  vllm框架2组       swift balancer 策略
│
├── kv_offload/
│   ├── __init__.py                       vllm框架2组
│   ├── cpu_npu.py                        vllm框架2组       KV cache CPU↔NPU 卸载
│   └── npu.py                            vllm框架2组       KV cache NPU 侧管理
│
├── lora/
│   ├── __init__.py                       vllm模型2组
│   ├── lora_ops.py                       vllm模型2组       LoRA 算子（sgmv/add）
│   ├── punica_npu.py                     vllm模型2组       punica NPU Multi-LoRA 实现
│   └── utils.py                          vllm模型2组       LoRA 工具函数
│
├── model_loader/
│   ├── __init__.py                       vllm模型2组
│   ├── netloader/
│   │   ├── __init__.py                   vllm模型2组
│   │   ├── load.py                       vllm模型2组       模型加载入口
│   │   ├── netloader.py                  vllm模型2组       网络加载器（PP 模型分发）
│   │   ├── utils.py                      vllm模型2组       netloader 工具
│   │   ├── executor/
│   │   │   ├── __init__.py               vllm模型2组
│   │   │   ├── elastic_load.py           vllm模型2组       弹性加载执行器（PP）
│   │   │   └── netloader_pg.py           vllm模型2组       netloader process group
│   │   └── interaction/
│   │       ├── __init__.py               vllm模型2组
│   │       └── elastic.py                vllm模型2组       弹性交互协议（PP）
│   └── rfork/
│       ├── __init__.py                   vllm模型2组
│       ├── rfork_loader.py               vllm模型2组       rfork 加载器（PP 权重分发）
│       ├── rfork_worker.py               vllm模型2组       rfork worker
│       ├── seed_protocol.py              vllm模型2组       rfork 种子文件协议
│       ├── seed_server.py                vllm模型2组       rfork 种子文件 server
│       └── transfer_backend.py           vllm模型2组       rfork 传输后端
│
├── ops/
│   ├── __init__.py                       公共              算子模块入口
│   ├── activation.py                     vllm模型1组 / vllm框架1组   激活函数算子（SiLU/GELU 等）
│   ├── conv.py                           vllm框架2组       卷积算子（长序列 causal conv）
│   ├── flashcomm2_oshard_manager.py      vllm框架3组       flashcomm2 O shard 管理
│   ├── gdn.py                            vllm框架2组 / vllm框架1组   GDN 门控增量网络算子
│   ├── layer_shard_linear.py             vllm模型1组       DP 层切分线性层
│   ├── layernorm.py                      vllm模型1组 / vllm框架1组   layernorm/rmsnorm 算子
│   ├── linear.py                         vllm模型1组       TP 线性层
│   ├── linear_op.py                      vllm模型1组       线性层底层实现
│   ├── mla.py                            vllm框架1组       MLA 算子
│   ├── mm_encoder_attention.py           公共              多模态 encoder attention
│   ├── qwen2_decoder.py                  公共              qwen2 decoder 专用算子
│   ├── register_custom_ops.py            公共              自定义算子注册
│   ├── rel_pos_attention.py              公共              相对位置 attention
│   ├── rotary_embedding.py               vllm模型1组 / vllm框架1组   RoPE 旋转位置编码
│   ├── vocab_parallel_embedding.py       vllm模型1组       TP 词表并行 embedding
│   ├── weight_prefetch.py                vllm框架1组       权重预取（CPU→NPU 异步搬运）
│   ├── fused_moe/
│   │   ├── __init__.py                   vllm框架3组
│   │   ├── comm_utils.py                 vllm框架3组       MoE 通信工具函数
│   │   ├── experts_selector.py           vllm框架3组       MoE experts 选择器
│   │   ├── fused_moe.py                  vllm框架3组       fused MoE 前向主逻辑
│   │   ├── moe_comm_method.py            vllm框架3组       MoE 通信方式（flashcomm/TP）
│   │   ├── moe_mlp.py                    vllm框架3组       MoE MLP 层
│   │   ├── moe_runtime_args.py           vllm框架3组       MoE 运行时参数
│   │   ├── moe_stage_contracts.py        vllm框架3组       MoE stage 契约
│   │   ├── moe_stage_params.py           vllm框架3组       MoE stage 参数
│   │   ├── prepare_finalize.py           vllm框架3组       MoE prepare/finalize
│   │   └── token_dispatcher.py           vllm框架3组       MoE token 分发与负载调度
│   └── triton/
│       ├── __init__.py                   公共              triton 算子入口
│       ├── batch_memcpy.py               vllm框架1组       批量 memcpy（triton）
│       ├── bincount.py                   vllm框架1组 / vllm框架2组   bincount 计数（triton）
│       ├── fused_gdn_gating.py           vllm框架2组       GDN fused gating（triton）
│       ├── gdn_chunk_meta.py             vllm框架2组       GDN chunk meta 计算（triton）
│       ├── layernorm_gated.py            vllm模型1组 / vllm框架1组   gated layernorm（triton）
│       ├── muls_add.py                   公共              mul+add 融合（triton）
│       ├── penalty.py                    vllm框架2组       采样惩罚项（triton）
│       ├── reject_sample.py              vllm框架2组       投机解码 reject sample（triton）
│       ├── rope.py                       vllm模型1组 / vllm框架1组   RoPE（triton）
│       ├── triton_utils.py               公共              triton 工具函数
│       ├── activation/
│       │   ├── __init__.py               vllm模型1组
│       │   └── swiglu_quant.py           vllm模型1组       SwiGLU + 量化融合（triton）
│       ├── batch_invariant/
│       │   ├── __init__.py               vllm框架1组
│       │   ├── matmul.py                 vllm框架1组       批量不变 matmul（triton，chunk prefill）
│       │   ├── mean.py                   vllm框架1组       批量不变 mean（triton）
│       │   ├── rmsnorm.py                vllm框架1组       批量不变 rmsnorm（triton）
│       │   └── softmax.py                vllm框架1组       批量不变 softmax（triton）
│       ├── fla/
│       │   ├── __init__.py               vllm框架2组
│       │   ├── chunk.py                  vllm框架2组       FLA chunk 主逻辑（长序列）
│       │   ├── chunk_delta_h.py          vllm框架2组       FLA delta_h 计算（长序列）
│       │   ├── chunk_delta_hupdate.py    vllm框架2组       FLA delta_h 更新（长序列）
│       │   ├── chunk_o.py                vllm框架2组       FLA chunk output（长序列）
│       │   ├── chunk_o_update.py         vllm框架2组       FLA chunk o 更新（长序列）
│       │   ├── chunk_scaled_dot_kkt.py   vllm框架2组       FLA chunk scaled dot KKT（长序列）
│       │   ├── cumsum.py                 vllm框架2组       FLA cumsum（长序列）
│       │   ├── fused_qkvzba_split_reshape.py  vllm框架2组   FLA fused QKV 拆分重塑（长序列）
│       │   ├── l2norm.py                 vllm框架2组       FLA L2 norm（长序列）
│       │   ├── layernorm_guard.py        vllm框架2组       FLA layernorm guard 保护值（长序列）
│       │   ├── sigmoid_gating.py         vllm框架2组       FLA sigmoid gating（长序列）
│       │   ├── solve_tril.py             vllm框架2组       FLA solve tril（长序列）
│       │   ├── utils.py                  vllm框架2组       FLA 工具函数
│       │   └── wy_fast.py                vllm框架2组       FLA wy fast 算法（长序列）
│       ├── linearnorm/
│       │   ├── __init__.py               vllm模型1组
│       │   ├── split_qkv_rmsnorm_mrope.py      vllm模型1组    QKV split+rmsnorm+mrope 融合（TP 友好）
│       │   ├── split_qkv_rmsnorm_rope.py       vllm模型1组    QKV split+rmsnorm+rope 融合（TP 友好）
│       │   └── split_qkv_tp_rmsnorm_rope.py    vllm模型1组    QKV split+rmsnorm+rope 融合（TP 变体）
│       ├── mamba/
│       │   ├── __init__.py               vllm框架2组
│       │   ├── causal_conv1d.py          vllm框架2组       causal conv1d（triton，长序列）
│       │   ├── lightning_attn.py         vllm框架2组       lightning attention（triton，长序列）
│       │   └── linear_attn.py            vllm框架2组       linear attention（triton，长序列）
│       └── spec_decode/
│           ├── __init__.py               vllm框架2组
│           └── utils.py                  vllm框架2组       投机解码 triton 工具
│
├── patch/
│   ├── __init__.py                       公共              monkey patch 入口
│   ├── platform/
│   │   ├── __init__.py                   公共
│   │   ├── patch_balance_schedule.py     vllm框架3组       负载均衡调度 patch
│   │   ├── patch_distributed.py          vllm模型1组 / vllm模型2组   分布式 patch（DP/TP/EP 配置）
│   │   ├── patch_kv_cache_interface.py   vllm框架2组       KV cache 接口 patch
│   │   ├── patch_mamba_config.py         vllm框架2组       mamba 模型配置 patch（长序列）
│   │   ├── patch_mamba_config_310.py     边端推理组       310P mamba 配置 patch
│   │   ├── patch_minimax_m2_config.py    vllm框架2组       minimax m2 配置 patch（长序列）
│   │   ├── patch_multiproc_executor.py   公共              多进程执行器 patch
│   │   ├── patch_profiling_chunk.py      vllm框架1组       profiling chunk patch
│   │   ├── patch_sched_yield.py          vllm框架3组       调度 yield patch
│   │   └── patch_torch_accelerator.py    vllm模型1组 / vllm模型2组   torch 加速器 patch
│   └── worker/
│       ├── __init__.py                   公共
│       ├── _hccl_pg_registry.py          vllm框架3组       HCCL process group 注册表
│       ├── patch_bailing_moe_linear.py   vllm框架3组       bailing MoE linear patch（flashcomm）
│       ├── patch_bert.py                 vllm框架2组       bert 模型 patch（长序列）
│       ├── patch_cudagraph.py            vllm框架1组       cudagraph patch（ACL graph 相关）
│       ├── patch_deepseek_mtp.py         vllm框架2组       deepseek mtp patch（投机解码）
│       ├── patch_distributed.py          vllm模型1组 / vllm模型2组   分布式 worker patch
│       ├── patch_draft_quarot.py         vllm框架2组       draft quarot patch（投机解码量化）
│       ├── patch_gdn_attn.py             vllm框架2组       GDN attention patch（长序列）
│       ├── patch_gqa_c8.py               vllm模型1组       GQA C8 patch（量化）
│       ├── patch_huanyuan_vl.py          vllm模型1组       huanyuan VL 模型 patch
│       ├── patch_kimi_k25.py             vllm模型2组       kimi k25 模型 patch
│       ├── patch_mamba_utils.py          vllm框架2组       mamba 工具 patch（长序列）
│       ├── patch_minimax_m2.py           vllm框架2组       minimax m2 模型 patch（长序列）
│       ├── patch_minimax_m2_linear_attn.py  vllm框架2组    minimax m2 linear attention patch（长序列）
│       ├── patch_module.py               vllm模型2组       module patch（PP/LoRA）
│       ├── patch_multimodal_merge.py     vllm模型1组       多模态 merge patch
│       ├── patch_npugraph_ex_triton.py   vllm框架1组       npu graph + triton 扩展 patch
│       ├── patch_qwen3_5.py              vllm模型2组       qwen3.5 模型 patch
│       ├── patch_qwen3_dflash.py         vllm模型1组       qwen3 dflash patch
│       ├── patch_qwen3_next_mtp.py       vllm框架2组       qwen3 next mtp patch（投机）
│       ├── patch_qwen3vl.py              vllm模型1组       qwen3vl 模型 patch
│       ├── patch_rejection_sampler.py    vllm框架2组       rejection sampler patch（投机）
│       ├── patch_routed_experts_capturer.py  vllm框架2组    routed experts capturer patch（EPLB）
│       ├── patch_triton.py               vllm模型2组       triton patch
│       └── patch_weight_utils.py         vllm模型1组       权重工具 patch（量化/并行）
│
├── sample/
│   ├── __init__.py                       vllm框架1组
│   ├── penalties.py                      vllm框架1组       采样惩罚项（frequency/repetition 等）
│   ├── rejection_sampler.py              vllm框架1组       rejection sampler（投机解码采样验证）
│   └── sampler.py                        vllm框架1组       采样器主逻辑（topk/topp）
│
├── spec_decode/
│   ├── __init__.py                       vllm框架2组
│   ├── dflash_proposer.py                vllm框架2组       dflash proposer（投机解码）
│   ├── draft_proposer.py                 vllm框架2组       draft proposer 基类
│   ├── eagle_proposer.py                 vllm框架2组       eagle proposer（投机解码）
│   ├── medusa_proposer.py                vllm框架2组       medusa proposer（投机解码）
│   ├── ngram_proposer.py                 vllm框架2组       ngram proposer（投机解码）
│   ├── suffix_proposer.py                vllm框架2组       suffix proposer（投机解码）
│   └── utils.py                          vllm框架2组       投机解码工具函数
│
├── quantization/
│   ├── __init__.py                       vllm模型1组
│   ├── compressed_tensors_config.py      vllm模型1组       压缩张量配置
│   ├── method_adapters.py                vllm模型1组       量化方法适配器
│   ├── modelslim_config.py               vllm模型1组       modelslim 量化配置
│   ├── quant_parser.py                   vllm模型1组       量化参数解析
│   ├── quant_type.py                     vllm模型1组       量化类型定义
│   ├── utils.py                          vllm模型1组       量化工具函数
│   └── methods/
│       ├── __init__.py                   vllm模型1组
│       ├── base.py                       vllm模型1组       量化方法基类
│       ├── kv_c8.py                      vllm模型1组       KV cache int8 量化
│       ├── registry.py                   vllm模型1组       量化方法注册表
│       ├── w4a16.py                      vllm模型1组       权重量化 int4 / 激活 fp16
│       ├── w4a4_flatquant.py             vllm模型1组       w4a4 flatquant 量化
│       ├── w4a4_laos_dynamic.py          vllm模型1组       w4a4 laos 动态量化
│       ├── w4a4_mxfp4.py                 vllm模型1组       w4a4 mxfp4 量化
│       ├── w4a8.py                       vllm模型1组       权重量化 int4 / 激活 int8
│       ├── w8a16.py                      vllm模型1组       权重量化 int8 / 激活 fp16
│       ├── w8a8_dynamic.py               vllm模型1组       w8a8 动态量化
│       ├── w8a8_mxfp8.py                 vllm模型1组       w8a8 mxfp8 量化
│       ├── w8a8_pdmix.py                 vllm模型1组       w8a8 pdmix 量化
│       └── w8a8_static.py                vllm模型1组       w8a8 静态量化
│
├── worker/
│   ├── __init__.py                       公共              worker 模块入口
│   ├── block_table.py                    vllm框架1组       block table 管理（KV cache 块映射）
│   ├── kvcomp_utils.py                   vllm框架2组       KV 压缩工具
│   ├── model_runner_v1.py                vllm框架1组       v1 model runner（控制模型执行流程）
│   ├── npu_input_batch.py                vllm框架1组       NPU 输入 batch 准备（prefill/decode）
│   ├── pcp_utils.py                      vllm框架1组       PCP prefix cache prefill 工具
│   ├── worker.py                         公共              worker 主入口
│   └── v2/
│       ├── __init__.py                   vllm框架1组
│       ├── aclgraph_utils.py             vllm框架1组       v2 ACL graph 工具
│       ├── attn_utils.py                 vllm框架1组       v2 attention 工具
│       ├── block_table.py                vllm框架1组       v2 block table
│       ├── input_batch.py                vllm框架1组       v2 输入 batch
│       ├── model_runner.py               vllm框架1组       v2 model runner（核心 attn+graph 调度）
│       ├── states.py                     vllm框架1组       v2 状态管理
│       ├── utils.py                      vllm框架1组       v2 工具函数
│       ├── model_states/
│       │   ├── __init__.py               vllm模型1组 / vllm模型2组
│       │   └── default.py                vllm模型1组 / vllm模型2组   默认模型状态管理
│       ├── sample/
│       │   ├── __init__.py               vllm框架1组
│       │   ├── bad_words.py              vllm框架1组       v2 bad words 过滤
│       │   ├── gumbel.py                 vllm框架1组       v2 gumbel 采样
│       │   ├── logprob.py                vllm框架1组       v2 logprob 计算
│       │   ├── min_p.py                  vllm框架1组       v2 min_p 采样
│       │   └── penalties.py              vllm框架1组       v2 采样惩罚项
│       └── spec_decode/
│           └── eagle/
│               ├── __init__.py           vllm框架2组
│               ├── aclgraph.py           vllm框架2组       v2 eagle ACL graph
│               └── speculator.py         vllm框架2组       v2 eagle speculator
│
├── xlite/
│   ├── __init__.py                       公共              轻量推理入口
│   ├── utils.py                          公共              xlite 工具
│   ├── xlite.py                          公共              xlite 主逻辑
│   ├── xlite_model_runner.py             公共              xlite model runner
│   └── xlite_worker.py                   公共              xlite worker
│
└── _310p/
    ├── __init__.py                       边端推理组       310P 平台入口
    ├── block_table.py                    边端推理组       310P block table
    ├── model_runner_310p.py              边端推理组       310P model runner
    ├── npu_input_batch.py                边端推理组       310P NPU 输入 batch
    ├── sharded_state_loader_310p.py      边端推理组       310P 分片状态加载
    ├── worker_310p.py                    边端推理组       310P worker
    ├── attention/
    │   ├── __init__.py                   边端推理组
    │   ├── attention_mask.py             边端推理组       310P attention mask
    │   ├── attention_v1.py               边端推理组       310P attention 实现
    │   └── metadata_builder.py           边端推理组       310P 元数据构建
    ├── fused_moe/
    │   ├── __init__.py                   边端推理组
    │   ├── experts_selector.py           边端推理组       310P MoE experts 选择器
    │   ├── fused_moe.py                  边端推理组       310P fused MoE 实现
    │   ├── moe_comm_method.py            边端推理组       310P MoE 通信方式
    │   ├── moe_mlp.py                    边端推理组       310P MoE MLP
    │   └── token_dispatcher.py           边端推理组       310P MoE token 分发
    ├── ops/
    │   ├── __init__.py                   边端推理组
    │   ├── activation.py                 边端推理组       310P 激活函数算子
    │   ├── causal_conv1d.py              边端推理组       310P causal conv1d
    │   ├── conv.py                       边端推理组       310P 卷积算子
    │   ├── layernorm.py                  边端推理组       310P layernorm 算子
    │   ├── mm_encoder_attention.py       边端推理组       310P 多模态 encoder attention
    │   ├── rotary_embedding.py           边端推理组       310P RoPE
    │   ├── vocab_parallel_embedding.py   边端推理组       310P TP 词表并行 embedding
    │   └── fla/
    │       ├── __init__.py               边端推理组
    │       ├── chunk_gated_delta_rule.py 边端推理组       310P chunk gated delta rule
    │       ├── fused_gdn_gating.py       边端推理组       310P fused GDN gating
    │       ├── fused_recurrent_gated_delta_rule.py  边端推理组   310P fused recurrent gated delta rule
    │       └── gdn_310.py                边端推理组       310P GDN 实现
    ├── quantization/
    │   ├── __init__.py                   边端推理组
    │   ├── modelslim_config.py           边端推理组       310P modelslim 量化配置
    │   └── methods/
    │       ├── __init__.py               边端推理组
    │       ├── registry.py               边端推理组       310P 量化方法注册
    │       ├── w8a8_dynamic.py           边端推理组       310P w8a8 动态量化
    │       ├── w8a8_static.py            边端推理组       310P w8a8 静态量化
    │       ├── w8a8s.py                  边端推理组       310P w8a8s 量化
    │       └── w8a8sc.py                 边端推理组       310P w8a8sc 量化
    └── sample/
        ├── __init__.py                   边端推理组
        └── sampler.py                    边端推理组       310P 采样器
```

---

## csrc/

```
csrc/
├── CMakeLists.txt                        公共              构建入口
│
├── aclnn_torch_adapter/                  vllm框架1组       ACL 与 PyTorch 适配层
├── add_rms_norm_bias/                    vllm框架1组       attention 残差+rmsnorm+bias 融合算子
├── apply_top_k_top_p_custom/             vllm框架1组       采样 topk/topp 自定义算子
├── batch_matmul_transpose/               vllm框架1组       attention 批量 matmul 转置算子
├── mla_preprocess/                       vllm框架1组       MLA attention 预处理算子
├── sparse_flash_attention/               vllm框架1组       sparse flash attention 自定义算子
│
├── causal_conv1d/                        vllm框架2组       长序列 causal conv1d（CANN 算子）
├── causal_conv1d_v310/                   边端推理组       310P causal conv1d
├── copy_and_expand_eagle_inputs/         vllm框架2组       eagle 投机解码输入扩展算子
├── hamming_dist_top_k/                   vllm框架2组       投机解码 hamming 距离 topk 算子
├── lightning_indexer_quant/              vllm框架2组       长序列 lightning indexer 量化算子
├── lightning_indexer_vllm/               vllm框架2组       长序列 lightning indexer vllm 算子
├── recurrent_gated_delta_rule/           vllm框架2组       长序列 FLA 递归门控增量规则算子
├── recurrent_gated_delta_rule_v310/      边端推理组       310P 递归门控增量规则算子
├── reshape_and_cache_bnsd/               vllm框架2组       KV cache reshape & cache 算子
├── transpose_kv_cache_by_block/          vllm框架2组       KV cache block 转置算子
│
├── dispatch_ffn_combine/                 vllm框架3组       MoE dispatch+ffn+combine 三合一融合算子
├── dispatch_ffn_combine_bf16/            vllm框架3组       MoE 三合一融合算子 bf16 版本
├── dispatch_ffn_combine_w4_a8/           vllm框架3组       MoE 三合一融合算子 w4a8 量化版本
├── dispatch_gmm_combine_decode/          vllm框架3组       decode 阶段 GMM combine 算子
├── dispatch_layout/                      vllm框架3组       dispatch 布局转换算子
├── moe_combine_normal/                   vllm框架3组       MoE combine 算子
├── moe_dispatch_normal/                  vllm框架3组       MoE dispatch 算子
├── moe_gating_top_k/                     vllm框架3组       MoE gating top-k 算子
├── moe_grouped_matmul/                   vllm框架3组       MoE grouped matmul 算子
├── moe_init_routing_custom/              vllm框架3组       MoE routing 初始化算子
├── notify_dispatch/                      vllm框架3组       调度通知 dispatch 算子
│
├── grouped_matmul_swiglu_quant_weight_nz_tensor_list/  vllm模型1组   量化分组 matmul+swiglu 融合算子
├── matmul_allreduce_add_rmsnorm/         vllm模型1组       TP matmul+allreduce+rmsnorm 三合一融合算子
│
├── kernels/                              公共              通用 kernel 基础库
├── utils/                                公共              通用工具（inc/ 头文件, src/ 源文件）
├── cmake/                                公共              构建系统脚本
└── third_party/                          公共              第三方依赖（catlass 等）
```

---

## 按组汇总

### vllm框架1组 — CPU优化 / ChunkPrefill / PD分离 / 池化 / Attn Backend / ACL Graph

```
vllm_ascend/
├── ascend_forward_context.py            前向传播上下文（chunk prefill）
├── batch_invariant.py                   chunk prefill 批量不变性分析
├── cpu_binding.py                       CPU 亲和性绑定
├── profiling_config.py                  profiling chunk 配置
├── attention/ (除 kvcomp_attn/)         attention backend
├── compilation/ (除 sequence_parallelism*)   ACL graph 编译
├── core/                                调度核心
├── device_allocator/                    内存池化
├── distributed/kv_transfer/             PD 分离 / KV 传输
├── ops/mla.py                           MLA 算子
├── ops/weight_prefetch.py               权重预取
├── ops/triton/batch_memcpy.py           批量 memcpy
├── ops/triton/batch_invariant/          chunk prefill 批量不变算子
├── sample/                              采样
├── worker/ (除 kvcomp_utils.py, spec_decode/)   worker v1
├── worker/v2/ (除 model_states/, spec_decode/)  worker v2 核心
├── patch/platform/patch_profiling_chunk.py
├── patch/worker/patch_cudagraph.py
└── patch/worker/patch_npugraph_ex_triton.py

csrc/
├── aclnn_torch_adapter/
├── add_rms_norm_bias/
├── apply_top_k_top_p_custom/
├── batch_matmul_transpose/
├── mla_preprocess/
└── sparse_flash_attention/
```

### vllm框架2组 — 长序列 / 投机解码 / EPLB / KV Cache

```
vllm_ascend/
├── attention/kvcomp_attn/               KV 压缩 attention
├── eplb/                                expert parallel load balancing
├── kv_offload/                          KV cache 卸载
├── spec_decode/                         投机解码
├── ops/conv.py                          卷积算子
├── ops/gdn.py                           GDN 算子
├── ops/triton/fused_gdn_gating.py
├── ops/triton/gdn_chunk_meta.py
├── ops/triton/penalty.py
├── ops/triton/reject_sample.py
├── ops/triton/fla/                      长序列 FLA
├── ops/triton/mamba/                    长序列 mamba/linear attention
├── ops/triton/spec_decode/             投机解码 triton 工具
├── worker/kvcomp_utils.py
├── worker/v2/spec_decode/eagle/        v2 eagle 投机
├── patch/platform/patch_kv_cache_interface.py
├── patch/platform/patch_mamba_config.py
├── patch/platform/patch_minimax_m2_config.py
├── patch/worker/patch_bert.py
├── patch/worker/patch_deepseek_mtp.py
├── patch/worker/patch_draft_quarot.py
├── patch/worker/patch_gdn_attn.py
├── patch/worker/patch_mamba_utils.py
├── patch/worker/patch_minimax_m2.py
├── patch/worker/patch_minimax_m2_linear_attn.py
├── patch/worker/patch_qwen3_next_mtp.py
├── patch/worker/patch_rejection_sampler.py
└── patch/worker/patch_routed_experts_capturer.py

csrc/
├── causal_conv1d/
├── copy_and_expand_eagle_inputs/
├── hamming_dist_top_k/
├── lightning_indexer_quant/
├── lightning_indexer_vllm/
├── recurrent_gated_delta_rule/
├── reshape_and_cache_bnsd/
└── transpose_kv_cache_by_block/
```

### vllm框架3组 — 调度 / FlashComm / RL 场景

```
vllm_ascend/
├── flash_common3_context.py             flashcomm 上下文
├── distributed/device_communicators/    设备通信（HCCL）
├── ops/flashcomm2_oshard_manager.py     flashcomm2 O shard 管理
├── ops/fused_moe/                       fused MoE（含调度/通信）
├── patch/platform/patch_balance_schedule.py   负载均衡调度
├── patch/platform/patch_sched_yield.py        调度 yield
├── patch/worker/_hccl_pg_registry.py          HCCL PG 注册表
└── patch/worker/patch_bailing_moe_linear.py   bailing MoE linear

csrc/
├── dispatch_ffn_combine/                MoE 三合一融合算子
├── dispatch_ffn_combine_bf16/           MoE 三合一 bf16
├── dispatch_ffn_combine_w4_a8/          MoE 三合一 w4a8
├── dispatch_gmm_combine_decode/         decode GMM combine
├── dispatch_layout/                     dispatch 布局转换
├── moe_combine_normal/                  MoE combine
├── moe_dispatch_normal/                 MoE dispatch
├── moe_gating_top_k/                    MoE gating
├── moe_grouped_matmul/                  MoE grouped matmul
├── moe_init_routing_custom/             MoE routing 初始化
└── notify_dispatch/                     调度通知
```

### vllm模型1组 — 模型侧并行(DP/TP/EP) / 量化 / Function Call

```
vllm_ascend/
├── meta_registration.py                 模型元信息注册
├── distributed/parallel_state.py        并行状态管理
├── distributed/utils.py                 分布式工具
├── ops/activation.py                    激活函数
├── ops/layer_shard_linear.py            DP 层切分
├── ops/layernorm.py                     layernorm
├── ops/linear.py                        TP 线性层
├── ops/linear_op.py                     线性层底层实现
├── ops/rotary_embedding.py              RoPE
├── ops/vocab_parallel_embedding.py      TP 词表并行
├── ops/triton/activation/swiglu_quant.py   SwiGLU+量化融合
├── ops/triton/layernorm_gated.py        gated layernorm
├── ops/triton/rope.py                   RoPE (triton)
├── ops/triton/linearnorm/               TP QKV 融合算子
├── quantization/                        量化方法
├── worker/v2/model_states/              模型状态管理
├── patch/platform/patch_distributed.py
├── patch/platform/patch_torch_accelerator.py
├── patch/worker/patch_distributed.py
├── patch/worker/patch_gqa_c8.py         GQA C8 量化
├── patch/worker/patch_huanyuan_vl.py
├── patch/worker/patch_multimodal_merge.py
├── patch/worker/patch_qwen3_dflash.py
├── patch/worker/patch_qwen3vl.py
└── patch/worker/patch_weight_utils.py

csrc/
├── grouped_matmul_swiglu_quant_weight_nz_tensor_list/
└── matmul_allreduce_add_rmsnorm/
```

### vllm模型2组 (= vllm模型1组全部 + PP并行 / Multi-LoRA)

```
vllm_ascend/
├── (vllm模型1组全部文件)
├── compilation/passes/sequence_parallelism.py    序列并行 pass
├── compilation/passes/sequence_parallelism_moe.py   序列并行 MoE pass
├── lora/                                Multi-LoRA
├── model_loader/                        模型加载（PP 权重分发）
├── patch/worker/patch_kimi_k25.py
├── patch/worker/patch_module.py         module patch (PP/LoRA)
├── patch/worker/patch_qwen3_5.py
└── patch/worker/patch_triton.py
```

### 边端推理组 — 310P 平台专属

```
vllm_ascend/_310p/                       310P 全量代码
├── block_table.py
├── model_runner_310p.py
├── npu_input_batch.py
├── sharded_state_loader_310p.py
├── worker_310p.py
├── attention/                           310P attention
├── fused_moe/                           310P fused MoE
├── ops/                                 310P 算子
├── quantization/                        310P 量化
└── sample/                              310P 采样

vllm_ascend/patch/platform/patch_mamba_config_310.py

csrc/
├── causal_conv1d_v310/                  310P causal conv1d
└── recurrent_gated_delta_rule_v310/     310P 递归门控增量规则
```
