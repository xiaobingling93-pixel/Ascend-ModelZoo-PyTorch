/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "operations/fusion/linear/linear.h"
#include "operations/fusion/linear/linear_parallel.h"
#include "operations/fusion/norm/norm_linear.h"
#include "operations/fusion/mlp/mlp.h"
#include "operations/fusion/moe/sparse_moe.h"
#include "operations/fusion/moe/moe_shared_expert.h"
#include "models/deepseekv2/operation/latent_attention.h"
#include "models/deepseekv2/layer/decoder_layer.h"

namespace atb_speed {
namespace deepseekV2 {

std::map<std::string, std::vector<std::string>> GetDeepseekV2LayerInTensorCandidates()
{
    std::map<std::string, std::vector<std::string>> deepseekV2LayerInTensorCandidates = {
        {"default", {
            "in_hidden_states", "in_expert_array", "in_expert_group", "in_one_hot", "in_zero_hot",
            "in_final_state",
            "in_cos_table", "in_sin_table", "in_attention_mask", "in_k_cache", "in_v_cache", "in_seq_len",
            "in_place_holder", "in_token_offset", "in_layer_id", "in_block_tables", "in_slots"}},
        {"default_weight", {
            "in_input_norm_weight", "in_input_norm_bias", "in_input_norm_new_weight", "in_input_norm_new_bias",
            "in_q_proj_a_weight", "in_q_proj_a_bias", "in_q_proj_a_descale", "in_q_proj_a_offset", "in_q_proj_a_scale",
            "in_q_proj_a_compress_idx", "in_q_proj_a_layernorm_weight", "in_q_proj_a_layernorm_bias",
            "in_q_proj_b_weight", "in_q_proj_b_bias", "in_q_proj_b_descale", "in_q_proj_b_offset", "in_q_proj_b_scale",
            "in_q_proj_b_compress_idx", "in_kv_proj_with_mqa_weight", "in_kv_proj_with_mqa_bias",
            "in_kv_proj_with_mqa_descale", "in_kv_proj_with_mqa_offset", "in_kv_proj_with_mqa_scale",
            "in_kv_proj_with_mqa_compress_idx", "in_kv_proj_a_layernorm_weight", "in_kv_proj_a_layernorm_bias",
            "in_k_proj_b_for_q_weight", "in_k_proj_b_for_q_bias", "in_k_proj_b_for_q_descale",
            "in_k_proj_b_for_q_offset", "in_k_proj_b_for_q_scale", "in_k_proj_b_for_q_compress_idx",
            "in_v_proj_b_for_o_weight", "in_v_proj_b_for_o_bias", "in_v_proj_b_for_o_descale",
            "in_v_proj_b_for_o_offset", "in_v_proj_b_for_o_scale", "in_v_proj_b_for_o_compress_idx",
            "in_attention_out_weight", "in_attention_out_bias", "in_attention_out_descale", "in_attention_out_offset",
            "in_attention_out_scale", "in_attention_out_compress_idx", "in_selfattention_out_norm_weight",
            "in_selfattention_out_norm_bias", "in_selfattention_out_new_norm_weight",
            "in_selfattention_out_new_norm_bias", "in_mlp_gateup_weight_shared_expert",
            "in_mlp_gateup_bias_shared_expert", "in_mlp_gateup_descale_shared_expert",
            "in_mlp_gateup_offset_shared_expert", "in_mlp_gateup_scale_shared_expert",
            "in_mlp_gateup_compress_idx_shared_expert", "in_mlp_down_weight_shared_expert",
            "in_mlp_down_bias_shared_expert", "in_mlp_down_descale_shared_expert",
            "in_mlp_down_offset_shared_expert", "in_mlp_down_scale_shared_expert",
            "in_mlp_down_compress_idx_shared_expert", "in_shared_expert_gate_weight", "in_shared_expert_gate_bias",
            "in_shared_expert_gate_descale", "in_shared_expert_gate_offset", "in_shared_expert_gate_scale",
            "in_shared_expert_gate_compress_idx", "in_block_sparse_moe_gate_weight", "in_block_sparse_moe_gate_bias",
            "in_block_sparse_moe_gate_descale", "in_block_sparse_moe_gate_offset", "in_block_sparse_moe_gate_scale",
            "in_block_sparse_moe_gate_compress_idx", "in_mlp_gateup_weight_expert", "in_mlp_gateup_bias_expert",
            "in_mlp_gateup_descale_expert", "in_mlp_gateup_offset_expert", "in_mlp_gateup_scale_expert",
            "in_mlp_gateup_compress_idx_expert", "in_mlp_down_weight_expert", "in_mlp_down_bias_expert",
            "in_mlp_down_descale_expert", "in_mlp_down_offset_expert", "in_mlp_down_scale_expert",
            "in_mlp_down_compress_idx_expert"}},
        {"attn_dp", {
            "in_shard_effective_token_indices", "in_token_index_with_padding", "in_skip_padding_token_indices"}},
        {"ep", {
            "in_start_expert_idx", "in_device_expert_count"}},
        {"attn_tp_for_dynamic_ep", {
            "in_attention_padding_idx", "in_attention_unpadding_idx"}},
        {"dynamic_ep", {
            "in_lty_idx", "in_moe_idx"}},
    };
    return deepseekV2LayerInTensorCandidates;
}

std::map<std::string, std::vector<std::string>> GetDeepseekV2LayerIntermediateTensorCandidates()
{
    std::map<std::string, std::vector<std::string>> deepseekV2LayerIntermediateTensorCandidates = {
        {"default", {
            "intermediate_attention_out", "intermediate_selfattention_norm_out",
            "intermediate_moe_out_with_shared"}},
        {"shared_expert", {
            "intermediate_shared_expert_out"}},
        {"attn_dp", {
            "intermediate_mlp_out", "intermediate_dp_attn_out_with_padding",
            "intermediate_dp_attn_out_all_with_padding",
            "intermediate_dp_attn_out_all", "intermediate_dp_mlp_out"}},
        {"attn_dp_last_layer", {
            "intermediate_mlp_out", "intermediate_dp_attn_out_with_padding",
            "intermediate_dp_attn_out_all_with_padding",
            "intermediate_dp_attn_out_all"}},
        {"attn_tp", {
            "intermediate_mlp_out"}},
        {"dynamic_ep", {
            "intermediate_attention_out_padding", "intermediate_a2a_out", "intermediate_attention_out_scatter",
            "intermediate_hidden_states_padding", "intermediate_hidden_states_scatter", "intermediate_mlp_out_all"}},
        {"dynamic_ep_last_layer", {
            "intermediate_layer_out", "intermediate_layer_out_with_padding",
            "intermediate_layer_out_all_with_padding"}},
    };
    return deepseekV2LayerIntermediateTensorCandidates;
}

atb::Status ConstructMoeEpTensorMap(
    const DecoderLayerParam &param,
    std::map<std::string, std::vector<std::string>> deepseekV2InTensorCandidates,
    std::map<std::string, std::vector<std::string>> deepseekV2IntermediateCandidates,
    std::vector<std::string> &inTensorList,
    std::vector<std::string> &intermediateTensorList)
{
    atb_speed::common::AddTensorToList(deepseekV2InTensorCandidates, "ep", inTensorList);
    if (param.isDynamicEp) {
        if (param.hasAttnTp) {
            atb_speed::common::AddTensorToList(deepseekV2InTensorCandidates,
                "attn_tp_for_dynamic_ep", inTensorList);
            if (!param.isDenseLayer) {
                atb_speed::common::AddTensorToList(deepseekV2IntermediateCandidates,
                    "dynamic_ep", intermediateTensorList);
                atb_speed::common::AddTensorToList(deepseekV2IntermediateCandidates,
                    "attn_tp", intermediateTensorList);
            }
        }
        if (param.hasAttnDp && param.hasMoeEp && param.isLastLayer) {
            atb_speed::common::AddTensorToList(deepseekV2IntermediateCandidates,
                "dynamic_ep_last_layer", intermediateTensorList);
        }
        atb_speed::common::AddTensorToList(deepseekV2InTensorCandidates, "dynamic_ep", inTensorList);
    }
    return atb::NO_ERROR;
}

std::map<std::string, uint32_t> ConstructTensorMap(
    const DecoderLayerParam &param, uint32_t &inTensorNum, uint32_t &outTensorNum, uint32_t &internalTensorNum)
{
    auto deepseekV2InTensorCandidates = GetDeepseekV2LayerInTensorCandidates();
    auto deepseekV2IntermediateCandidates = GetDeepseekV2LayerIntermediateTensorCandidates();
    std::vector<std::string> inTensorList = {};
    std::vector<std::string> intermediateTensorList = {};
    std::vector<std::string> outTensorList = {"out_decoder_layer"};

    atb_speed::common::AddTensorToList(deepseekV2InTensorCandidates, "default_weight", inTensorList);
    atb_speed::common::AddTensorToList(deepseekV2InTensorCandidates, "default", inTensorList);
    atb_speed::common::AddTensorToList(deepseekV2IntermediateCandidates, "default", intermediateTensorList);

    if (param.hasSharedExpert && !param.isDenseLayer) {
        atb_speed::common::AddTensorToList(deepseekV2IntermediateCandidates, "shared_expert", intermediateTensorList);
    }

    if (param.hasAttnDp) {
        atb_speed::common::AddTensorToList(deepseekV2InTensorCandidates, "attn_dp", inTensorList);
        if (param.isDenseLayer || (!param.isDynamicEp && \
            param.layerId < param.numHiddenLayers -1)) {
            atb_speed::common::AddTensorToList(deepseekV2IntermediateCandidates, "attn_dp", intermediateTensorList);
        } else if (param.expertParallelDegree !=2 && param.isLastLayer) { // 2: 动态ep
            atb_speed::common::AddTensorToList(deepseekV2IntermediateCandidates,
                                               "attn_dp_last_layer", intermediateTensorList);
        }
    } else if ((param.hasAttnTp && !param.isDynamicEp) || param.isDenseLayer) {
        if (param.worldSize > 1) {
            atb_speed::common::AddTensorToList(deepseekV2IntermediateCandidates, "attn_tp", intermediateTensorList);
        }
    }

    if (param.hasMoeEp) {
        ConstructMoeEpTensorMap(param, deepseekV2InTensorCandidates, deepseekV2IntermediateCandidates,
            inTensorList, intermediateTensorList);
    }

    inTensorNum = inTensorList.size();
    internalTensorNum = intermediateTensorList.size();
    outTensorNum = outTensorList.size();

    return atb_speed::common::GetTensorMap(inTensorList, outTensorList, intermediateTensorList);
}

atb::Status SetLatentAttentionParam(
    atb_speed::common::LatentAttentionParam<atb::infer::RmsNormParam> &latentAttentionParam,
    const DecoderLayerParam &param)
{
    latentAttentionParam.isGroupedQueryAttention = param.numAttentionHeadsPerRank != param.numKeyValueHeadsPerRank;
    latentAttentionParam.isBF16 = param.isBF16;
    latentAttentionParam.attnLinearQuantType = param.attnLinearQuantType;
    latentAttentionParam.packQuantType = param.packQuantType.at(0);
    latentAttentionParam.attnLinearTransposeType = param.attnLinearTransposeType;
    latentAttentionParam.enableLcoc = param.enableLcoc;
    atb::infer::RmsNormParam attenRmsNormParam;
    attenRmsNormParam.layerType = atb::infer::RmsNormParam::RmsNormType::RMS_NORM_NORM;
    attenRmsNormParam.normParam.epsilon = param.normEps;
    latentAttentionParam.normParamType = attenRmsNormParam;
    atb::infer::RmsNormParam attenRmsNormQuantParam;
    attenRmsNormQuantParam.layerType = atb::infer::RmsNormParam::RmsNormType::RMS_NORM_NORM;
    attenRmsNormQuantParam.normParam.epsilon = param.normEps;
    attenRmsNormQuantParam.normParam.quantType = atb::infer::QUANT_INT8;
    latentAttentionParam.normQuantParamType = attenRmsNormQuantParam;
    latentAttentionParam.qLoraRank = param.qLoraRank;
    latentAttentionParam.headNum = param.headNum;
    latentAttentionParam.qkNopeHeadDim = param.qkNopeHeadDim;
    latentAttentionParam.qkRopeHeadDim = param.qkRopeHeadDim;
    latentAttentionParam.kvLoraRank = param.kvLoraRank;
    latentAttentionParam.rotaryType = atb_speed::common::RotaryType::ALL_ROTARY;
    latentAttentionParam.ropeParam.rotaryCoeff = 2; // 2:设置新张量形状
    latentAttentionParam.isFA = param.isFA;
    latentAttentionParam.isPrefill = param.isPrefill;
    latentAttentionParam.headDim = param.hiddenSizePerAttentionHead;
    latentAttentionParam.selfAttentionParam.headNum = param.numAttentionHeadsPerRank;
    latentAttentionParam.selfAttentionParam.kvHeadNum = param.numAttentionHeadsPerRank;
    CHECK_PARAM_GT(param.hiddenSizePerAttentionHead, 0);
    latentAttentionParam.selfAttentionParam.qkScale = param.softmaxScale;
    latentAttentionParam.selfAttentionParam.isTriuMask = param.isPrefill ? 1 : 0;
    if (param.isFA) {
        latentAttentionParam.selfAttentionParam.calcType = param.isPrefill ? \
            atb::infer::SelfAttentionParam::CalcType::ENCODER : atb::infer::SelfAttentionParam::CalcType::DECODER;
    } else {
        latentAttentionParam.selfAttentionParam.calcType = atb::infer::SelfAttentionParam::CalcType::PA_ENCODER;
    }
    latentAttentionParam.selfAttentionParam.maskType = atb::infer::SelfAttentionParam::MaskType::MASK_TYPE_NORM;
    latentAttentionParam.pageAttentionParam.headNum = param.numAttentionHeadsPerRank;
    latentAttentionParam.pageAttentionParam.kvHeadNum = 1;
    latentAttentionParam.pageAttentionParam.mlaVHeadSize = param.kvLoraRank;
    latentAttentionParam.pageAttentionParam.qkScale = param.softmaxScale;
    latentAttentionParam.pageAttentionParam.maskType = atb::infer::PagedAttentionParam::MaskType::UNDEFINED;
    if (!param.isDynamicEp || param.isDenseLayer) {
        latentAttentionParam.selfOutLinearTensorParallelInfo = {
            param.attnTpRank, param.attnTpSize, param.backend, param.attnTpRankTableFile, nullptr,
            param.backend == "hccl" ? param.attnTpDomain : ""};
    }
    latentAttentionParam.reshapeCacheParm.kvCacheCfg = atb::infer::ReshapeAndCacheParam::KvCacheCfg::K_CACHE_V_BYPASS;
    return atb::NO_ERROR;
}

int64_t SetAttention(atb::GraphParam &opGraph, const DecoderLayerParam &param,
                     std::map<std::string, uint32_t> &tensorMap)
{
    atb::Node attentionNode;
    atb_speed::common::LatentAttentionParam<atb::infer::RmsNormParam> latentAttentionParam;
    SetLatentAttentionParam(latentAttentionParam, param);
    CHECK_OPERATION_STATUS_RETURN(Attention(latentAttentionParam, &attentionNode.operation));
    std::vector<std::string> attnInTensorNames = {
        "in_hidden_states",
        "in_input_norm_weight", "in_input_norm_bias", "in_input_norm_new_weight", "in_input_norm_new_bias",
        "in_q_proj_a_weight", "in_q_proj_a_bias", "in_q_proj_a_descale",
        "in_q_proj_a_offset", "in_q_proj_a_scale", "in_q_proj_a_compress_idx",
        "in_q_proj_a_layernorm_weight", "in_q_proj_a_layernorm_bias",
        "in_q_proj_b_weight", "in_q_proj_b_bias", "in_q_proj_b_descale",
        "in_q_proj_b_offset", "in_q_proj_b_scale", "in_q_proj_b_compress_idx",
        "in_kv_proj_with_mqa_weight", "in_kv_proj_with_mqa_bias", "in_kv_proj_with_mqa_descale",
        "in_kv_proj_with_mqa_offset", "in_kv_proj_with_mqa_scale", "in_kv_proj_with_mqa_compress_idx",
        "in_kv_proj_a_layernorm_weight", "in_kv_proj_a_layernorm_bias",
        "in_k_proj_b_for_q_weight", "in_k_proj_b_for_q_bias", "in_k_proj_b_for_q_descale",
        "in_k_proj_b_for_q_offset", "in_k_proj_b_for_q_scale", "in_k_proj_b_for_q_compress_idx",
        "in_v_proj_b_for_o_weight", "in_v_proj_b_for_o_bias", "in_v_proj_b_for_o_descale",
        "in_v_proj_b_for_o_offset", "in_v_proj_b_for_o_scale", "in_v_proj_b_for_o_compress_idx",
        "in_attention_out_weight", "in_attention_out_bias", "in_attention_out_descale",
        "in_attention_out_offset", "in_attention_out_scale", "in_attention_out_compress_idx",
        "in_cos_table", "in_sin_table", "in_seq_len", "in_k_cache",
        "in_attention_mask", "in_token_offset", "in_layer_id", "in_block_tables", "in_slots"
    };
    attentionNode.inTensorIds = atb_speed::common::GetTensorIdxList(tensorMap, attnInTensorNames);
    attentionNode.outTensorIds = atb_speed::common::GetTensorIdxList(tensorMap, {"intermediate_attention_out"});
    opGraph.nodes.push_back(attentionNode);
    ATB_SPEED_LOG_DEBUG("Attention calculation success");
    return atb::NO_ERROR;
}

atb::Status SetPadding(atb::GraphParam &opGraph, std::map<std::string, uint32_t> tensorMap)
{
    atb::Node gatherNode;
    atb::infer::GatherParam gatherParam;
    CHECK_OPERATION_STATUS_RETURN(CreateOperation(gatherParam, &gatherNode.operation));

    gatherNode.inTensorIds = atb_speed::common::GetTensorIdxList(tensorMap, {"intermediate_attention_out",
                                                                             "in_attention_padding_idx"});
    gatherNode.outTensorIds = atb_speed::common::GetTensorIdxList(tensorMap, {"intermediate_attention_out_padding"});

    opGraph.nodes.push_back(gatherNode);
    ATB_SPEED_LOG_DEBUG("create SetPadding");
    return atb::NO_ERROR;
}

atb::Status SetAllToALL(atb::GraphParam &opGraph, const DecoderLayerParam &param,
    std::map<std::string, uint32_t> tensorMap)
{
    atb::Node allToAllNode;
    atb::infer::AllToAllParam allToAllParam;

    allToAllParam.rank = param.attnTpRank;
    allToAllParam.rankSize = param.attnTpSize;
    allToAllParam.backend = param.backend;
    allToAllParam.commDomain = param.attnTpDomain;
    allToAllParam.rankTableFile = param.attnTpRankTableFile;

    CreateOperation(allToAllParam, &allToAllNode.operation);

    allToAllNode.inTensorIds = atb_speed::common::GetTensorIdxList(tensorMap, {"intermediate_attention_out_padding"});
    allToAllNode.outTensorIds =  atb_speed::common::GetTensorIdxList(tensorMap, {"intermediate_a2a_out"});

    allToAllNode.inTensorReshapeFuncs.resize(allToAllNode.inTensorIds.size());
    allToAllNode.inTensorReshapeFuncs[0] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
            newShape.dimNum = 2; // 2: dimNum
            newShape.dims[0] = oldShape.dims[0];
            newShape.dims[1] = oldShape.dims[1];
    };

    opGraph.nodes.push_back(allToAllNode);
    return atb::NO_ERROR;
}

atb::Status SetReduce(atb::GraphParam &opGraph, const DecoderLayerParam &param,
    std::map<std::string, uint32_t> tensorMap)
{
    atb::Node reduceNode;
    atb::infer::ReduceParam reduceParam;

    reduceParam.reduceType = atb::infer::ReduceParam::ReduceType::REDUCE_SUM;
    reduceParam.axis.resize(1); // 调整 SVector 的大小
    reduceParam.axis[0] = 0; // 将第一个元素设置为 1

    CreateOperation(reduceParam, &reduceNode.operation);

    reduceNode.inTensorIds = atb_speed::common::GetTensorIdxList(tensorMap, {"intermediate_a2a_out"});
    reduceNode.outTensorIds = atb_speed::common::GetTensorIdxList(tensorMap, {"intermediate_attention_out_scatter"});
    reduceNode.inTensorReshapeFuncs.resize(reduceNode.inTensorIds.size());
    reduceNode.inTensorReshapeFuncs[0] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = 3; // dimNum: 3, [tp_size, bs * seq_len, hidden_size]
        newShape.dims[0] = param.attnTpSize;
        newShape.dims[1] = oldShape.dims[0] / param.attnTpSize;
        newShape.dims[2] = oldShape.dims[1]; // 2: dim 2
    };
    opGraph.nodes.push_back(reduceNode);
    return atb::NO_ERROR;
}

atb::Status SetTPReduceScatter(atb::GraphParam &opGraph, const DecoderLayerParam &param,
    std::map<std::string, uint32_t> tensorMap)
{
    CHECK_OPERATION_STATUS_RETURN(SetAllToALL(opGraph, param, tensorMap));
    CHECK_OPERATION_STATUS_RETURN(SetReduce(opGraph, param, tensorMap));
    return atb::NO_ERROR;
}

atb::Status SetResidualPadding(atb::GraphParam &opGraph, std::map<std::string, uint32_t> tensorMap)
{
    atb::Node gatherNode;
    atb::infer::GatherParam gatherParam;
    CHECK_OPERATION_STATUS_RETURN(CreateOperation(gatherParam, &gatherNode.operation));

    gatherNode.inTensorIds = atb_speed::common::GetTensorIdxList(tensorMap, {"in_hidden_states",
                                                                             "in_attention_padding_idx"});
    gatherNode.outTensorIds = atb_speed::common::GetTensorIdxList(tensorMap, {"intermediate_hidden_states_padding"});
    opGraph.nodes.push_back(gatherNode);
    return atb::NO_ERROR;
}

atb::Status SetResidualSliceNode(atb::GraphParam &opGraph, const DecoderLayerParam &param,
    std::map<std::string, uint32_t> tensorMap)
{
    atb::infer::SliceParam sliceParam;
    atb::Node sliceNode;

    sliceParam.offsets.resize(3); // 3: Slice offset dim
    sliceParam.offsets[0] = param.attnTpRank;
    sliceParam.offsets[1] = 0;
    sliceParam.offsets[2] = 0; // 2: dim：2

    sliceParam.size.resize(3); // 3: Slice Size dim
    sliceParam.size[0] = 1;
    sliceParam.size[1] = -1;
    sliceParam.size[2] = -1; // 2: dim：2
    CreateOperation(sliceParam, &sliceNode.operation);

    sliceNode.inTensorIds = atb_speed::common::GetTensorIdxList(tensorMap, {"intermediate_hidden_states_padding"});
    sliceNode.outTensorIds = atb_speed::common::GetTensorIdxList(tensorMap, {"intermediate_hidden_states_scatter"});

    sliceNode.inTensorReshapeFuncs.resize(sliceNode.inTensorIds.size());
    sliceNode.inTensorReshapeFuncs[0] = [=] (const atb::Dims &oldShape, atb::Dims &newShape) {
        if (oldShape.dimNum == 2) { // 2: dimNum
            newShape.dimNum = 3; // 3: dimNum
            newShape.dims[0] = param.attnTpSize;
            newShape.dims[1] = oldShape.dims[0] / param.attnTpSize;
            newShape.dims[2] = oldShape.dims[1]; // 2: dim 2
        } else {
            newShape.dimNum = 3; // 3: dimNum
            newShape.dims[0] = param.attnTpSize;
            newShape.dims[1] = oldShape.dims[0]*oldShape.dims[1] / param.attnTpSize;
            newShape.dims[2] = oldShape.dims[2]; // 2: dim 2
        }
    };
    opGraph.nodes.push_back(sliceNode);
    return atb::NO_ERROR;
}

atb::Status SetSelfResidualAdd(atb::GraphParam &opGraph, const DecoderLayerParam &param,
    std::map<std::string, uint32_t> &tensorMap)
{
    atb::Node selfResidualAddNode;
    atb::infer::ElewiseParam addParam;
    addParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD;
    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(addParam, &selfResidualAddNode.operation));
    if (param.hasAttnTp && param.isDynamicEp && !param.isDenseLayer) {
        selfResidualAddNode.inTensorIds = \
        atb_speed::common::GetTensorIdxList(tensorMap, {"intermediate_hidden_states_scatter",
                                                        "intermediate_attention_out_scatter"});
        selfResidualAddNode.outTensorIds = atb_speed::common::GetTensorIdxList(tensorMap,
            {"intermediate_attention_out_scatter"});
        selfResidualAddNode.inTensorReshapeFuncs.resize(selfResidualAddNode.inTensorIds.size());
        selfResidualAddNode.inTensorReshapeFuncs[0] = [=] (const atb::Dims &oldShape, atb::Dims &newShape) {
            newShape.dimNum = 2; // 2: dimNum
            newShape.dims[0] = oldShape.dims[0] * oldShape.dims[1];
            newShape.dims[1] = oldShape.dims[2]; // 2: dim 2
        };
    } else {
        selfResidualAddNode.inTensorIds = \
        atb_speed::common::GetTensorIdxList(tensorMap, {"in_hidden_states", "intermediate_attention_out"});
        selfResidualAddNode.outTensorIds = atb_speed::common::GetTensorIdxList(tensorMap,
            {"intermediate_attention_out"});
    }

    opGraph.nodes.push_back(selfResidualAddNode);
    ATB_SPEED_LOG_DEBUG("SelfResidualAdd calculation success");
    return atb::NO_ERROR;
}

int64_t SetAllGather(atb::GraphParam &opGraph, const DecoderLayerParam &param,
                     std::map<std::string, uint32_t> &tensorMap)
{
    atb::Node padNode;
    atb::infer::GatherParam padParam;
    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(padParam, &padNode.operation));
    padNode.inTensorIds = atb_speed::common::GetTensorIdxList(
        tensorMap, {"intermediate_attention_out", "in_token_index_with_padding"});
    padNode.outTensorIds = atb_speed::common::GetTensorIdxList(tensorMap, {"intermediate_dp_attn_out_with_padding"});
    opGraph.nodes.push_back(padNode);

    atb::Node allGatherNode;
    atb::infer::AllGatherParam allGatherParam;
    allGatherParam.rank = param.attnDpRank;
    allGatherParam.rankSize = param.attnDpSize;
    allGatherParam.backend = param.backend;
    allGatherParam.rankTableFile = param.attnDpRankTableFile;
    if (param.backend == "hccl") {
        allGatherParam.commDomain = param.attnDpDomain;
    }
    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(allGatherParam, &allGatherNode.operation));
    allGatherNode.inTensorIds = atb_speed::common::GetTensorIdxList(
        tensorMap, {"intermediate_dp_attn_out_with_padding"});
    allGatherNode.outTensorIds = atb_speed::common::GetTensorIdxList(
        tensorMap, {"intermediate_dp_attn_out_all_with_padding"});
    opGraph.nodes.push_back(allGatherNode);

    atb::Node unpadNode;
    atb::infer::GatherParam unpadParam;
    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(unpadParam, &unpadNode.operation));
    unpadNode.inTensorIds = atb_speed::common::GetTensorIdxList(
        tensorMap, {"intermediate_dp_attn_out_all_with_padding", "in_skip_padding_token_indices"});
    unpadNode.outTensorIds = atb_speed::common::GetTensorIdxList(tensorMap, {"intermediate_dp_attn_out_all"});
    unpadNode.inTensorReshapeFuncs.reserve(unpadNode.inTensorIds.size());
    unpadNode.inTensorReshapeFuncs.resize(unpadNode.inTensorIds.size());
    unpadNode.inTensorReshapeFuncs[0] = [=] (const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = 2; // 2：新shape维度为2
        newShape.dims[0] = oldShape.dims[0] * oldShape.dims[1]; // 0, 0, 1： 新shape前两维合轴
        newShape.dims[1] = oldShape.dims[2]; // 1, 2: 新shape最后一维不变
    };
    opGraph.nodes.push_back(unpadNode);

    ATB_SPEED_LOG_DEBUG("AllGather calculation success");
    return atb::NO_ERROR;
}

atb::Status SetNormQauntInTensors(
    std::vector<std::string> &selfNormInTensorNames,
    atb::infer::RmsNormParam &mlpRmsNormParam,
    atb::infer::RmsNormParam &mlpRmsNormQuantParam,
    const DecoderLayerParam &param,
    atb::Node &selfNormNode)
{
    if (param.mlpNormQuantType == atb::infer::QUANT_INT8) { // w8a8
        CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(mlpRmsNormQuantParam, &selfNormNode.operation));
        if (param.isAntiOutlier) {
            selfNormInTensorNames.push_back("in_selfattention_out_new_norm_weight");
            selfNormInTensorNames.push_back("in_selfattention_out_new_norm_bias");
        } else {
            selfNormInTensorNames.push_back("in_selfattention_out_norm_weight");
            selfNormInTensorNames.push_back("in_selfattention_out_norm_bias");
        }
    } else if (param.normHasBias) { // FP
        CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(mlpRmsNormParam, &selfNormNode.operation));
        selfNormInTensorNames.push_back("in_selfattention_out_norm_weight");
        selfNormInTensorNames.push_back("in_selfattention_out_new_norm_bias");
    } else {
        CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(mlpRmsNormParam, &selfNormNode.operation));
        if (param.isAntiOutlier) {
            selfNormInTensorNames.push_back("in_selfattention_out_new_norm_weight");
        } else {
            selfNormInTensorNames.push_back("in_selfattention_out_norm_weight");
        }
    }
    return atb::NO_ERROR;
}

int64_t SetSelfNorm(atb::GraphParam &opGraph, const DecoderLayerParam &param,
                    std::map<std::string, uint32_t> &tensorMap)
{
    atb::Node selfNormNode;
    atb::infer::RmsNormParam mlpRmsNormParam;
    atb::infer::RmsNormParam mlpRmsNormQuantParam;
    std::vector<std::string> selfNormInTensorNames;
    std::vector<std::string> selfNormOutTensorNames;
    selfNormOutTensorNames.push_back("intermediate_selfattention_norm_out");
    if (param.enableAddNorm) {
        mlpRmsNormParam.layerType = atb::infer::RmsNormParam::RmsNormType::RMS_NORM_PRENORM;
        mlpRmsNormParam.preNormParam.epsilon = param.normEps;
        mlpRmsNormQuantParam.layerType = atb::infer::RmsNormParam::RmsNormType::RMS_NORM_PRENORM;
        mlpRmsNormQuantParam.preNormParam.epsilon = param.normEps;
        mlpRmsNormQuantParam.preNormParam.quantType = atb::infer::QUANT_INT8;
        if (param.hasAttnTp && param.expertParallelDegree == 2 && !param.isDenseLayer) { // 2: Dynamic EP
            selfNormInTensorNames.push_back("intermediate_hidden_states_scatter");
            selfNormOutTensorNames.push_back("intermediate_attention_out_scatter");
        } else {
            selfNormInTensorNames.push_back("in_hidden_states");
            selfNormOutTensorNames.push_back("intermediate_attention_out");
        }
    } else {
        mlpRmsNormParam.layerType = atb::infer::RmsNormParam::RmsNormType::RMS_NORM_NORM;
        mlpRmsNormParam.normParam.epsilon = param.normEps;
        mlpRmsNormQuantParam.layerType = atb::infer::RmsNormParam::RmsNormType::RMS_NORM_NORM;
        mlpRmsNormQuantParam.normParam.epsilon = param.normEps;
        mlpRmsNormQuantParam.normParam.quantType = atb::infer::QUANT_INT8;
    }
    if (param.hasAttnTp && param.isDynamicEp && !param.isDenseLayer) {
        selfNormInTensorNames.push_back("intermediate_attention_out_scatter");
    } else {
        selfNormInTensorNames.push_back(
            param.hasAttnDp && (!param.isDynamicEp || param.isDenseLayer) ? \
            "intermediate_dp_attn_out_all" : "intermediate_attention_out");
    }
    SetNormQauntInTensors(selfNormInTensorNames, mlpRmsNormParam, mlpRmsNormQuantParam, param, selfNormNode);
    selfNormNode.inTensorIds = atb_speed::common::GetTensorIdxList(tensorMap, selfNormInTensorNames);
    selfNormNode.outTensorIds = atb_speed::common::GetTensorIdxList(tensorMap, selfNormOutTensorNames);
    opGraph.nodes.push_back(selfNormNode);
    ATB_SPEED_LOG_DEBUG("SelfNorm calculation success");
    return atb::NO_ERROR;
}

int64_t SetMlpExpert(atb::GraphParam &opGraph, const DecoderLayerParam &param,
                     std::map<std::string, uint32_t> &tensorMap)
{
    atb::Node mlpExpertNode;
    atb_speed::common::SharedExpertParam mlpExpertParam;
    mlpExpertParam.isBF16 = param.isBF16;
    mlpExpertParam.transposeGateup = param.transpose;
    mlpExpertParam.transposeDown = param.transpose;
    mlpExpertParam.hasSharedExpertGate = false;
    mlpExpertParam.mlpLinearQuantType = param.mlpLinearQuantType;
    mlpExpertParam.mlpLinearTransposeType = param.mlpLinearTransposeType;
    mlpExpertParam.packQuantType = param.packQuantType.at(1);
    atb_speed::common::CreateSharedExpertOperation(mlpExpertParam, &mlpExpertNode.operation);
    std::vector<std::string> mlpExpertInTensorNames = {
        "intermediate_selfattention_norm_out",
        "in_mlp_gateup_weight_shared_expert", "in_mlp_gateup_bias_shared_expert",
        "in_mlp_gateup_descale_shared_expert", "in_mlp_gateup_offset_shared_expert",
        "in_mlp_gateup_scale_shared_expert", "in_mlp_gateup_compress_idx_shared_expert",
        "in_mlp_down_weight_shared_expert", "in_mlp_down_bias_shared_expert",
        "in_mlp_down_descale_shared_expert", "in_mlp_down_offset_shared_expert",
        "in_mlp_down_scale_shared_expert", "in_mlp_down_compress_idx_shared_expert",
        "in_shared_expert_gate_weight", "in_shared_expert_gate_bias", "in_shared_expert_gate_descale",
        "in_shared_expert_gate_offset", "in_shared_expert_gate_scale", "in_shared_expert_gate_compress_idx"
    };
    mlpExpertNode.inTensorIds = atb_speed::common::GetTensorIdxList(tensorMap, mlpExpertInTensorNames);
    mlpExpertNode.outTensorIds = atb_speed::common::GetTensorIdxList(tensorMap, {"intermediate_moe_out_with_shared"});
    opGraph.nodes.push_back(mlpExpertNode);
    ATB_SPEED_LOG_DEBUG("mlp expert calculation success");
    return atb::NO_ERROR;
}

atb::Status SetSetMoeParam(atb_speed::common::SparseMoeParam &sparseMoeParam, const DecoderLayerParam &param)
{
    sparseMoeParam.isBF16 = param.isBF16;
    sparseMoeParam.transpose = param.transpose;
    sparseMoeParam.numOfExperts = param.numOfExperts;
    sparseMoeParam.numOfDeviceExperts = param.numOfDeviceExperts;
    sparseMoeParam.num = param.numOfSelectedExperts;
    sparseMoeParam.routingMethod = param.routingMethod;
    sparseMoeParam.numOfGroups = param.numOfGroups;
    sparseMoeParam.topkGroups = param.topkGroups;
    sparseMoeParam.expertParallelDegree = param.expertParallelDegree;
    sparseMoeParam.isDynamicEp = param.isDynamicEp;
    sparseMoeParam.deviceExpert = param.deviceExpert;
    sparseMoeParam.routedScalingFactor = param.routedScalingFactor;
    sparseMoeParam.processLogits = param.processLogits;
    sparseMoeParam.moeLinearQuantType = param.moeLinearQuantType;
    sparseMoeParam.packQuantType = param.packQuantType.at(1);
    sparseMoeParam.enableFusedRouting = param.enableFusedRouting;
    sparseMoeParam.backend = param.backend;
    sparseMoeParam.hasMoeEp = param.hasMoeEp;
    sparseMoeParam.moeEpRank = param.moeEpRank;
    sparseMoeParam.moeEpSize = param.moeEpSize;
    sparseMoeParam.moeEpDomain = param.moeEpDomain;
    sparseMoeParam.moeEpRankTableFile = param.moeEpRankTableFile;
    sparseMoeParam.hasMlpTp = param.hasMlpTp;
    sparseMoeParam.mlpTpRank = param.mlpTpRank;
    sparseMoeParam.mlpTpSize = param.mlpTpSize;
    sparseMoeParam.mlpTpDomain = param.mlpTpDomain;
    sparseMoeParam.mlpTpRankTableFile = param.mlpTpRankTableFile;
    return atb::NO_ERROR;
}

int64_t SetMoe(atb::GraphParam &opGraph, const DecoderLayerParam &param, std::map<std::string, uint32_t> tensorMap)
{
    atb::Node moeNode;
    atb_speed::common::SparseMoeParam sparseMoeParam;
    SetSetMoeParam(sparseMoeParam, param);

    atb_speed::common::CreateSparseMoeOperation(sparseMoeParam, &moeNode.operation);
    if (moeNode.operation == nullptr) {
        ATB_SPEED_LOG_ERROR("SparseMoe op is nullptr: ");
    }
    std::vector<std::string> moeInTensorNames = {
        "intermediate_selfattention_norm_out",
        "in_block_sparse_moe_gate_weight", "in_block_sparse_moe_gate_bias", "in_block_sparse_moe_gate_descale",
        "in_block_sparse_moe_gate_offset", "in_block_sparse_moe_gate_scale", "in_block_sparse_moe_gate_compress_idx",
        "in_mlp_gateup_weight_expert", "in_mlp_gateup_bias_expert", "in_mlp_gateup_descale_expert",
        "in_mlp_gateup_offset_expert", "in_mlp_gateup_scale_expert", "in_mlp_gateup_compress_idx_expert",
        "in_mlp_down_weight_expert", "in_mlp_down_bias_expert", "in_mlp_down_descale_expert",
        "in_mlp_down_offset_expert", "in_mlp_down_scale_expert", "in_mlp_down_compress_idx_expert",
        "in_expert_array", "in_expert_group", "in_one_hot", "in_zero_hot",
    };
    if (param.hasMoeEp) {
        moeInTensorNames.push_back("in_start_expert_idx");
        moeInTensorNames.push_back("in_device_expert_count");
        if (param.isDynamicEp) {
            moeInTensorNames.push_back("in_lty_idx");
            moeInTensorNames.push_back("in_moe_idx");
        }
    }
    moeNode.inTensorIds = atb_speed::common::GetTensorIdxList(tensorMap, moeInTensorNames);
    moeNode.outTensorIds = atb_speed::common::GetTensorIdxList(tensorMap, {"intermediate_moe_out_with_shared"});
    opGraph.nodes.push_back(moeNode);
    ATB_SPEED_LOG_DEBUG("Moe sparse calculation success");
    return atb::NO_ERROR;
}

int64_t SetSharedExpert(atb::GraphParam &opGraph, const DecoderLayerParam &param,
                        std::map<std::string, uint32_t> tensorMap)
{
    atb::Node sharedExpertNode;
    atb_speed::common::SharedExpertParam sharedExpertParam;
    sharedExpertParam.isBF16 = param.isBF16;
    sharedExpertParam.transposeGateup = param.transpose;
    sharedExpertParam.transposeDown = param.transpose;
    sharedExpertParam.hasSharedExpertGate = param.hasSharedExpertGate;
    sharedExpertParam.mlpLinearQuantType = param.mlpLinearQuantType;
    sharedExpertParam.mlpLinearTransposeType = param.mlpLinearTransposeType;
    sharedExpertParam.packQuantType = param.packQuantType.at(1);
    atb_speed::common::CreateSharedExpertOperation(sharedExpertParam, &sharedExpertNode.operation);
    std::vector<std::string> sharedExpertInTensorNames = {
        "intermediate_selfattention_norm_out",
        "in_mlp_gateup_weight_shared_expert", "in_mlp_gateup_bias_shared_expert",
        "in_mlp_gateup_descale_shared_expert", "in_mlp_gateup_offset_shared_expert",
        "in_mlp_gateup_scale_shared_expert", "in_mlp_gateup_compress_idx_shared_expert",
        "in_mlp_down_weight_shared_expert", "in_mlp_down_bias_shared_expert",
        "in_mlp_down_descale_shared_expert", "in_mlp_down_offset_shared_expert",
        "in_mlp_down_scale_shared_expert", "in_mlp_down_compress_idx_shared_expert",
        "in_shared_expert_gate_weight", "in_shared_expert_gate_bias", "in_shared_expert_gate_descale",
        "in_shared_expert_gate_offset", "in_shared_expert_gate_scale", "in_shared_expert_gate_compress_idx"
    };
    sharedExpertNode.inTensorIds = atb_speed::common::GetTensorIdxList(tensorMap, sharedExpertInTensorNames);
    sharedExpertNode.outTensorIds = atb_speed::common::GetTensorIdxList(tensorMap, {"intermediate_shared_expert_out"});
    opGraph.nodes.push_back(sharedExpertNode);
    ATB_SPEED_LOG_DEBUG("Shared expert calculation success");
    return atb::NO_ERROR;
}

int64_t AddExpertAdd(
    atb::GraphParam &opGraph,
    std::map<std::string, uint32_t> tensorMap)
{
    atb::Node expertAddNode;
    atb::infer::ElewiseParam addParam;
    addParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD;
    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(addParam, &expertAddNode.operation));
    expertAddNode.inTensorIds = atb_speed::common::GetTensorIdxList(tensorMap, {"intermediate_moe_out_with_shared",
                                                                                "intermediate_shared_expert_out"});
    expertAddNode.outTensorIds = atb_speed::common::GetTensorIdxList(tensorMap, {"intermediate_moe_out_with_shared"});
    opGraph.nodes.push_back(expertAddNode);
    ATB_SPEED_LOG_DEBUG("create add operation");
    return atb::NO_ERROR;
}

int64_t SetAllReduce(atb::GraphParam &opGraph, const DecoderLayerParam &param,
                     std::map<std::string, uint32_t> tensorMap)
{
    atb::Node moeAllReduceNode;
    atb::infer::AllReduceParam allReduceParam;
    allReduceParam.rank = param.mlpTpRank;
    allReduceParam.rankSize = param.mlpTpSize;
    allReduceParam.backend = param.backend;
    allReduceParam.rankTableFile = param.mlpTpRankTableFile;
    if (param.backend == "hccl") {
        allReduceParam.commDomain = param.mlpTpDomain;
    }
    CreateOperation(allReduceParam, &moeAllReduceNode.operation);
    if (moeAllReduceNode.operation == nullptr) {
        ATB_SPEED_LOG_ERROR("moeAllReduceNode op is nullptr: ");
    }
    moeAllReduceNode.inTensorIds = atb_speed::common::GetTensorIdxList(tensorMap, {"intermediate_moe_out_with_shared"});
    moeAllReduceNode.outTensorIds = atb_speed::common::GetTensorIdxList(tensorMap, {"intermediate_mlp_out"});
    opGraph.nodes.push_back(moeAllReduceNode);
    ATB_SPEED_LOG_DEBUG("create all reduce");
    return atb::NO_ERROR;
}

atb::Status SetMlpResidualAdd(atb::GraphParam &opGraph, const DecoderLayerParam &param,
                              std::map<std::string, uint32_t> tensorMap)
{
    atb::Node mlpResidualAddNode;
    atb::infer::ElewiseParam addParam;
    addParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD;
    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(addParam, &mlpResidualAddNode.operation));
    if (param.isDenseLayer || !param.isDynamicEp) {
        if (param.worldSize <= 1) {
            std::vector<std::string> mlpResidualAddInTensorNames = {
                param.hasAttnDp ? "intermediate_dp_attn_out_all" : "intermediate_attention_out",
                "intermediate_moe_out_with_shared"};
            mlpResidualAddNode.inTensorIds = atb_speed::common::GetTensorIdxList(
                tensorMap, mlpResidualAddInTensorNames);
        } else {
            std::vector<std::string> mlpResidualAddInTensorNames = {
                param.hasAttnDp ? "intermediate_dp_attn_out_all" : "intermediate_attention_out",
                "intermediate_mlp_out"};
            mlpResidualAddNode.inTensorIds = atb_speed::common::GetTensorIdxList(
                tensorMap, mlpResidualAddInTensorNames);
        }
        std::vector<std::string> mlpResidualAddOutTensorNames = {
            param.hasAttnDp && !param.isLastLayer ? \
            "intermediate_dp_mlp_out" : "out_decoder_layer"
        };
        mlpResidualAddNode.outTensorIds = atb_speed::common::GetTensorIdxList(tensorMap, mlpResidualAddOutTensorNames);
    } else {
        std::vector<std::string> mlpResidualAddInTensorNames = {
            param.hasAttnTp? "intermediate_attention_out_scatter":"intermediate_attention_out",
            "intermediate_moe_out_with_shared"
        };
        std::vector<std::string> mlpResidualAddOutTensorNames = {
            param.hasAttnTp ? "intermediate_mlp_out": (!param.hasAttnDp || !param.isLastLayer ? \
                "out_decoder_layer":"intermediate_layer_out")
        };
        mlpResidualAddNode.inTensorIds = atb_speed::common::GetTensorIdxList(tensorMap, mlpResidualAddInTensorNames);
        mlpResidualAddNode.outTensorIds = atb_speed::common::GetTensorIdxList(tensorMap, mlpResidualAddOutTensorNames);
    }
    opGraph.nodes.push_back(mlpResidualAddNode);
    ATB_SPEED_LOG_DEBUG("create mlpResidualAdd");
    return atb::NO_ERROR;
}

int64_t SetRevertAllGather(atb::GraphParam &opGraph, std::map<std::string, uint32_t> tensorMap)
{
    atb::Node revertAllGatherNode;
    atb::infer::GatherParam gatherParam;
    atb::CreateOperation(gatherParam, &revertAllGatherNode.operation);
    revertAllGatherNode.inTensorIds = atb_speed::common::GetTensorIdxList(
        tensorMap, {"intermediate_dp_mlp_out", "in_shard_effective_token_indices"});
    revertAllGatherNode.outTensorIds = atb_speed::common::GetTensorIdxList(tensorMap, {"out_decoder_layer"});
    opGraph.nodes.push_back(revertAllGatherNode);
    ATB_SPEED_LOG_DEBUG("create revertAllGatherNode");
    return atb::NO_ERROR;
}

atb::Status SetTPAllGatherNode(atb::GraphParam &opGraph, const DecoderLayerParam &param,
    std::map<std::string, uint32_t> tensorMap)
{
    atb::Node allGatherNode;
    atb::infer::AllGatherParam allGatherParam;

    allGatherParam.rank = param.attnTpRank;
    allGatherParam.rankSize = param.attnTpSize;
    allGatherParam.backend = param.backend;
    allGatherParam.commDomain = param.attnTpDomain;
    allGatherParam.rankTableFile = param.attnTpRankTableFile;

    CreateOperation(allGatherParam, &allGatherNode.operation);

    allGatherNode.inTensorIds = atb_speed::common::GetTensorIdxList(tensorMap, {"intermediate_mlp_out"});
    allGatherNode.outTensorIds = atb_speed::common::GetTensorIdxList(tensorMap, {"intermediate_mlp_out_all"});

    opGraph.nodes.push_back(allGatherNode);
    return atb::NO_ERROR;
}

atb::Status SetUnPadding(atb::GraphParam &opGraph, const DecoderLayerParam &param,
    std::map<std::string, uint32_t> tensorMap)
{
    atb::Node gatherNode;
    atb::infer::GatherParam gatherParam;
    CHECK_OPERATION_STATUS_RETURN(CreateOperation(gatherParam, &gatherNode.operation));

    gatherNode.inTensorIds = atb_speed::common::GetTensorIdxList(tensorMap, \
        {"intermediate_mlp_out_all", "in_attention_unpadding_idx"});
    gatherNode.outTensorIds = atb_speed::common::GetTensorIdxList(tensorMap, \
        {!param.hasAttnDp || !param.isLastLayer ? "out_decoder_layer":"intermediate_layer_out"});

    gatherNode.inTensorReshapeFuncs.resize(gatherNode.inTensorIds.size());
    gatherNode.inTensorReshapeFuncs[0] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
            newShape.dimNum = 2; // 2: dimNum
            newShape.dims[0] = oldShape.dims[0]*oldShape.dims[1];
            newShape.dims[1] =  oldShape.dims[2]; // 2: dim 2
    };
    opGraph.nodes.push_back(gatherNode);
    return atb::NO_ERROR;
}

int64_t SetLastLayerAllGather(atb::GraphParam &opGraph, const DecoderLayerParam &param,
    std::map<std::string, uint32_t> &tensorMap)
{
    atb::Node allGatherBeforeNode;
    atb::infer::GatherParam gatherParam0;
    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(gatherParam0, &allGatherBeforeNode.operation));
    allGatherBeforeNode.inTensorIds = atb_speed::common::GetTensorIdxList(
        tensorMap, {"intermediate_layer_out", "in_token_index_with_padding"});
    allGatherBeforeNode.outTensorIds = atb_speed::common::GetTensorIdxList(
        tensorMap, {"intermediate_layer_out_with_padding"});
    opGraph.nodes.push_back(allGatherBeforeNode);

    atb::Node allGatherNode;
    atb::infer::AllGatherParam allGatherParam;
    allGatherParam.rank = param.attnDpRank;
    allGatherParam.rankSize = param.attnDpSize;
    allGatherParam.backend = param.backend;
    allGatherParam.rankTableFile = param.attnDpRankTableFile;
    if (param.backend == "hccl") {
        allGatherParam.commDomain = param.attnDpDomain;
    }
    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(allGatherParam, &allGatherNode.operation));
    allGatherNode.inTensorIds = atb_speed::common::GetTensorIdxList(
        tensorMap, {"intermediate_layer_out_with_padding"});
    allGatherNode.outTensorIds = atb_speed::common::GetTensorIdxList(
        tensorMap, {"intermediate_layer_out_all_with_padding"});
    opGraph.nodes.push_back(allGatherNode);

    atb::Node allGatherAfterNode;
    atb::infer::GatherParam gatherParam1;
    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(gatherParam1, &allGatherAfterNode.operation));
    allGatherAfterNode.inTensorIds = atb_speed::common::GetTensorIdxList(
        tensorMap, {"intermediate_layer_out_all_with_padding", "in_skip_padding_token_indices"});
    allGatherAfterNode.outTensorIds = atb_speed::common::GetTensorIdxList(
        tensorMap, {"out_decoder_layer"});
    allGatherAfterNode.inTensorReshapeFuncs.reserve(allGatherAfterNode.inTensorIds.size());
    allGatherAfterNode.inTensorReshapeFuncs.resize(allGatherAfterNode.inTensorIds.size());
    allGatherAfterNode.inTensorReshapeFuncs[0] = [](const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = 2; // 2：新shape维度为2
        newShape.dims[0] = oldShape.dims[0] * oldShape.dims[1]; // 0, 0, 1： 新shape前两维合轴
        newShape.dims[1] = oldShape.dims[2]; // 1, 2: 新shape最后一维不变
    };
    opGraph.nodes.push_back(allGatherAfterNode);
    return atb::NO_ERROR;
}

atb::Status MoeLayer(std::map<std::string, uint32_t> &tensorMap,
    const DecoderLayerParam &param, atb::GraphParam &opGraph)
{
    if (param.isDynamicEp && !param.isDenseLayer) {
        CHECK_OPERATION_STATUS_RETURN(SetMlpResidualAdd(opGraph, param, tensorMap));
        if (param.hasAttnTp) {
            CHECK_OPERATION_STATUS_RETURN(SetTPAllGatherNode(opGraph, param, tensorMap));
            CHECK_OPERATION_STATUS_RETURN(SetUnPadding(opGraph, param, tensorMap));
        }
        if (param.hasAttnDp && param.isLastLayer) {
            CHECK_OPERATION_STATUS_RETURN(SetLastLayerAllGather(opGraph, param, tensorMap));
        }
    } else {
        if (param.worldSize > 1) {
            CHECK_OPERATION_STATUS_RETURN(SetAllReduce(opGraph, param, tensorMap));
        }
        CHECK_OPERATION_STATUS_RETURN(SetMlpResidualAdd(opGraph, param, tensorMap));
        if (param.hasAttnDp && !param.isLastLayer) {
            CHECK_OPERATION_STATUS_RETURN(SetRevertAllGather(opGraph, tensorMap));
        }
    }
    return atb::NO_ERROR;
}

atb::Status DecoderLayer(const DecoderLayerParam &param, atb::Operation **operation)
{
    atb::GraphParam opGraph;
    opGraph.name = param.isPrefill ? "Prefill_layer" : "Decoder_layer";
    std::map<std::string, uint32_t> tensorMap = ConstructTensorMap(
        param, opGraph.inTensorNum, opGraph.outTensorNum, opGraph.internalTensorNum);
    ATB_SPEED_LOG_DEBUG("layer graph inTensorNum: " << opGraph.inTensorNum);
    ATB_SPEED_LOG_DEBUG("layer graph outTensorNum: " << opGraph.outTensorNum);
    ATB_SPEED_LOG_DEBUG("layer graph internalTensorNum: " << opGraph.internalTensorNum);
    CHECK_OPERATION_STATUS_RETURN(SetAttention(opGraph, param, tensorMap));
    if (param.isDynamicEp && param.hasAttnTp && !param.isDenseLayer) {
        CHECK_OPERATION_STATUS_RETURN(SetPadding(opGraph, tensorMap)); // 0
        CHECK_OPERATION_STATUS_RETURN(SetTPReduceScatter(opGraph, param, tensorMap)); // 1-2
        CHECK_OPERATION_STATUS_RETURN(SetResidualPadding(opGraph, tensorMap)); // 3
        CHECK_OPERATION_STATUS_RETURN(SetResidualSliceNode(opGraph, param, tensorMap)); // 4
    }
    if (!param.enableAddNorm) {
        CHECK_OPERATION_STATUS_RETURN(SetSelfResidualAdd(opGraph, param, tensorMap));
    }
    if (param.hasAttnDp && !(param.isDynamicEp && !param.isDenseLayer)) {
        CHECK_OPERATION_STATUS_RETURN(SetAllGather(opGraph, param, tensorMap));
    }
    CHECK_OPERATION_STATUS_RETURN(SetSelfNorm(opGraph, param, tensorMap));
    if (param.isDenseLayer) {
        CHECK_OPERATION_STATUS_RETURN(SetMlpExpert(opGraph, param, tensorMap));
    } else {
        if (param.hasSharedExpert) {
            CHECK_OPERATION_STATUS_RETURN(SetSharedExpert(opGraph, param, tensorMap));
        }
        CHECK_OPERATION_STATUS_RETURN(SetMoe(opGraph, param, tensorMap));
        if (param.hasSharedExpert) {
            CHECK_OPERATION_STATUS_RETURN(AddExpertAdd(opGraph, tensorMap));
        }
    };
    MoeLayer(tensorMap, param, opGraph);
    opGraph.inferShapeFunc = [=] (const atb::SVector<atb::TensorDesc> &inTensorDescs,
                                 atb::SVector<atb::TensorDesc> &outTensorDescs) {
        if (param.hasAttnDp && param.isLastLayer) {
            outTensorDescs.at(0) = inTensorDescs.at(atb_speed::common::GetTensorIdx(tensorMap, "in_final_state"));
        } else {
            outTensorDescs.at(0) = inTensorDescs.at(0);
        }
        return atb::NO_ERROR;
    };
    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(opGraph, operation));
    return atb::NO_ERROR;
}

DecoderLayer::DecoderLayer() {}

DecoderLayer::~DecoderLayer() {}

} // namespace deepseekV2
} // namespace atb_speed
