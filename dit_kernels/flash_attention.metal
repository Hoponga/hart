#include <metal_stdlib>
using namespace metal;

// Constants for block size and memory management
constant int BLOCK_SIZE = 128;
constant int NUM_BLOCKS = 4;
constant float SCALE = 1.0 / sqrt(128.0); // Assuming head dimension of 128

// Forward pass kernel
kernel void flash_attention_forward(
    device const float* query [[buffer(0)]],
    device const float* key [[buffer(1)]],
    device const float* value [[buffer(2)]],
    device float* output [[buffer(3)]],
    device float* m [[buffer(4)]],  // For storing max values
    device float* l [[buffer(5)]],  // For storing sum of exponentials
    uint3 gid [[thread_position_in_grid]],
    uint3 tid [[thread_position_in_threadgroup]])
{
    // Get dimensions
    const int batch_size = gid.z;
    const int num_heads = gid.y;
    const int seq_len = gid.x;
    
    // Shared memory for block-level computation
    threadgroup float shared_mem[BLOCK_SIZE];
    threadgroup float shared_mem2[BLOCK_SIZE];
    
    // Initialize local variables
    float local_max = -INFINITY;
    float local_sum = 0.0;
    
    // Process blocks
    for (int b = 0; b < NUM_BLOCKS; b++) {
        // Load block of key-value pairs
        for (int i = 0; i < BLOCK_SIZE; i++) {
            int idx = b * BLOCK_SIZE + i;
            if (idx < seq_len) {
                // Compute attention scores
                float score = 0.0;
                for (int j = 0; j < 128; j++) { // Assuming head dimension of 128
                    score += query[batch_size * num_heads * seq_len * 128 + 
                                 num_heads * seq_len * 128 + 
                                 seq_len * 128 + j] *
                            key[batch_size * num_heads * seq_len * 128 + 
                                num_heads * seq_len * 128 + 
                                idx * 128 + j];
                }
                score *= SCALE;
                
                // Update local max and sum
                local_max = max(local_max, score);
                local_sum += exp(score - local_max);
            }
        }
        
        // Synchronize threads
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // Store block results
        if (tid.x == 0) {
            m[batch_size * num_heads * seq_len + num_heads * seq_len + seq_len] = local_max;
            l[batch_size * num_heads * seq_len + num_heads * seq_len + seq_len] = local_sum;
        }
    }
    
    // Compute final attention output
    float final_max = m[batch_size * num_heads * seq_len + num_heads * seq_len + seq_len];
    float final_sum = l[batch_size * num_heads * seq_len + num_heads * seq_len + seq_len];
    
    // Compute attention weights and apply to values
    for (int i = 0; i < seq_len; i++) {
        float score = 0.0;
        for (int j = 0; j < 128; j++) {
            score += query[batch_size * num_heads * seq_len * 128 + 
                         num_heads * seq_len * 128 + 
                         seq_len * 128 + j] *
                    key[batch_size * num_heads * seq_len * 128 + 
                        num_heads * seq_len * 128 + 
                        i * 128 + j];
        }
        score = exp(score * SCALE - final_max) / final_sum;
        
        // Apply attention weights to values
        for (int j = 0; j < 128; j++) {
            output[batch_size * num_heads * seq_len * 128 + 
                  num_heads * seq_len * 128 + 
                  seq_len * 128 + j] += score * 
                value[batch_size * num_heads * seq_len * 128 + 
                      num_heads * seq_len * 128 + 
                      i * 128 + j];
        }
    }
}

// Backward pass kernel
kernel void flash_attention_backward(
    device const float* query [[buffer(0)]],
    device const float* key [[buffer(1)]],
    device const float* value [[buffer(2)]],
    device const float* grad_output [[buffer(3)]],
    device float* grad_query [[buffer(4)]],
    device float* grad_key [[buffer(5)]],
    device float* grad_value [[buffer(6)]],
    device const float* m [[buffer(7)]],
    device const float* l [[buffer(8)]],
    uint3 gid [[thread_position_in_grid]],
    uint3 tid [[thread_position_in_threadgroup]])
{
    // Get dimensions
    const int batch_size = gid.z;
    const int num_heads = gid.y;
    const int seq_len = gid.x;
    
    // Shared memory for block-level computation
    threadgroup float shared_mem[BLOCK_SIZE];
    threadgroup float shared_mem2[BLOCK_SIZE];
    
    // Load max and sum values
    float final_max = m[batch_size * num_heads * seq_len + num_heads * seq_len + seq_len];
    float final_sum = l[batch_size * num_heads * seq_len + num_heads * seq_len + seq_len];
    
    // Compute gradients
    for (int i = 0; i < seq_len; i++) {
        float score = 0.0;
        for (int j = 0; j < 128; j++) {
            score += query[batch_size * num_heads * seq_len * 128 + 
                         num_heads * seq_len * 128 + 
                         seq_len * 128 + j] *
                    key[batch_size * num_heads * seq_len * 128 + 
                        num_heads * seq_len * 128 + 
                        i * 128 + j];
        }
        score = exp(score * SCALE - final_max) / final_sum;
        
        // Compute gradients for query, key, and value
        for (int j = 0; j < 128; j++) {
            float grad = grad_output[batch_size * num_heads * seq_len * 128 + 
                                   num_heads * seq_len * 128 + 
                                   seq_len * 128 + j];
            
            // Gradient for query
            grad_query[batch_size * num_heads * seq_len * 128 + 
                      num_heads * seq_len * 128 + 
                      seq_len * 128 + j] += score * grad * 
                key[batch_size * num_heads * seq_len * 128 + 
                    num_heads * seq_len * 128 + 
                    i * 128 + j];
            
            // Gradient for key
            grad_key[batch_size * num_heads * seq_len * 128 + 
                    num_heads * seq_len * 128 + 
                    i * 128 + j] += score * grad * 
                query[batch_size * num_heads * seq_len * 128 + 
                      num_heads * seq_len * 128 + 
                      seq_len * 128 + j];
            
            // Gradient for value
            grad_value[batch_size * num_heads * seq_len * 128 + 
                      num_heads * seq_len * 128 + 
                      i * 128 + j] += score * grad;
        }
    }
} 