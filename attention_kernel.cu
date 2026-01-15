#include <cuda_runtime.h>
#include <cmath>

// CUDA kernel for scaled dot-product attention
// Forward pass: computes Softmax(Q @ K^T / sqrt(d_k)) @ V
__global__ void scaled_dot_product_attention_forward(
    const float* Q,           // [batch_size, seq_len, d_k]
    const float* K,           // [batch_size, seq_len, d_k]
    const float* V,           // [batch_size, seq_len, d_v]
    float* attention_scores,  // [batch_size, seq_len, seq_len]
    float* output,            // [batch_size, seq_len, d_v]
    int batch_size,
    int seq_len,
    int d_k,
    int d_v,
    float scale
) {
    int batch_idx = blockIdx.z;
    int seq_idx_q = blockIdx.y;
    int feat_idx = threadIdx.x;
    
    // Compute attention scores: Q @ K^T / sqrt(d_k)
    float score = 0.0f;
    for (int i = 0; i < d_k; i++) {
        int q_idx = batch_idx * seq_len * d_k + seq_idx_q * d_k + i;
        for (int seq_idx_k = 0; seq_idx_k < seq_len; seq_idx_k++) {
            int k_idx = batch_idx * seq_len * d_k + seq_idx_k * d_k + i;
            score += Q[q_idx] * K[k_idx];
        }
    }
    
    int score_idx = batch_idx * seq_len * seq_len + seq_idx_q * seq_len + feat_idx;
    if (feat_idx < seq_len) {
        attention_scores[score_idx] = score * scale;
    }
}

// CUDA kernel for softmax over attention scores
__global__ void attention_softmax_kernel(
    float* attention_scores,  // [batch_size, seq_len, seq_len]
    int batch_size,
    int seq_len
) {
    int batch_idx = blockIdx.z;
    int seq_idx = blockIdx.y;
    int thread_idx = threadIdx.x;
    
    // Find max for numerical stability
    float max_val = -INFINITY;
    for (int i = thread_idx; i < seq_len; i += blockDim.x) {
        int idx = batch_idx * seq_len * seq_len + seq_idx * seq_len + i;
        max_val = fmaxf(max_val, attention_scores[idx]);
    }
    
    // Compute exp and sum
    float sum_exp = 0.0f;
    for (int i = thread_idx; i < seq_len; i += blockDim.x) {
        int idx = batch_idx * seq_len * seq_len + seq_idx * seq_len + i;
        attention_scores[idx] = expf(attention_scores[idx] - max_val);
        sum_exp += attention_scores[idx];
    }
    
    // Normalize
    for (int i = thread_idx; i < seq_len; i += blockDim.x) {
        int idx = batch_idx * seq_len * seq_len + seq_idx * seq_len + i;
        attention_scores[idx] /= (sum_exp + 1e-8f);
    }
}

// CUDA kernel for attention output: softmax(scores) @ V
__global__ void attention_output_kernel(
    const float* attention_scores,  // [batch_size, seq_len, seq_len]
    const float* V,                 // [batch_size, seq_len, d_v]
    float* output,                  // [batch_size, seq_len, d_v]
    int batch_size,
    int seq_len,
    int d_v
) {
    int batch_idx = blockIdx.z;
    int seq_idx_q = blockIdx.y;
    int feat_idx = threadIdx.x;
    
    if (feat_idx < d_v) {
        float output_val = 0.0f;
        for (int i = 0; i < seq_len; i++) {
            int score_idx = batch_idx * seq_len * seq_len + seq_idx_q * seq_len + i;
            int v_idx = batch_idx * seq_len * d_v + i * d_v + feat_idx;
            output_val += attention_scores[score_idx] * V[v_idx];
        }
        int out_idx = batch_idx * seq_len * d_v + seq_idx_q * d_v + feat_idx;
        output[out_idx] = output_val;
    }
}

// CUDA kernel for backward pass: gradient computation
__global__ void scaled_dot_product_attention_backward(
    const float* Q,                     // [batch_size, seq_len, d_k]
    const float* K,                     // [batch_size, seq_len, d_k]
    const float* V,                     // [batch_size, seq_len, d_v]
    const float* attention_scores,      // [batch_size, seq_len, seq_len]
    const float* grad_output,           // [batch_size, seq_len, d_v]
    float* grad_Q,                      // [batch_size, seq_len, d_k]
    float* grad_K,                      // [batch_size, seq_len, d_k]
    float* grad_V,                      // [batch_size, seq_len, d_v]
    float* grad_attention_scores,       // [batch_size, seq_len, seq_len]
    int batch_size,
    int seq_len,
    int d_k,
    int d_v,
    float scale
) {
    int batch_idx = blockIdx.z;
    int seq_idx = blockIdx.y;
    int feat_idx = threadIdx.x;
    
    // Gradient w.r.t. V
    if (feat_idx < d_v) {
        for (int i = 0; i < seq_len; i++) {
            int score_idx = batch_idx * seq_len * seq_len + seq_idx * seq_len + i;
            int grad_out_idx = batch_idx * seq_len * d_v + seq_idx * d_v + feat_idx;
            int grad_v_idx = batch_idx * seq_len * d_v + i * d_v + feat_idx;
            atomicAdd(&grad_V[grad_v_idx], attention_scores[score_idx] * grad_output[grad_out_idx]);
        }
    }
    
    // Gradient w.r.t. attention scores
    if (feat_idx < seq_len) {
        float grad_score = 0.0f;
        for (int i = 0; i < d_v; i++) {
            int grad_out_idx = batch_idx * seq_len * d_v + seq_idx * d_v + i;
            int v_idx = batch_idx * seq_len * d_v + feat_idx * d_v + i;
            grad_score += grad_output[grad_out_idx] * V[v_idx];
        }
        int score_idx = batch_idx * seq_len * seq_len + seq_idx * seq_len + feat_idx;
        grad_attention_scores[score_idx] = grad_score;
    }
}
