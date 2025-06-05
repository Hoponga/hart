import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math
from Metal import MTLCreateSystemDefaultDevice, MTLCompileOptions
import Metal

# todo: 
# is there an alternative to pyobjc bindings? 
class MetalFlashAttention(nn.Module):
    def __init__(self, device: torch.device):
        super().__init__()
        self.device = device
        
        # Load Metal shader
        self.mtl_device = MTLCreateSystemDefaultDevice()
        
        # Create compile options
        compile_options = MTLCompileOptions.alloc().init()
        compile_options.setFastMathEnabled_(True)
        
        # Load and compile the shader
        with open('flash_attention.metal', 'r') as f:
            source = f.read()
        
        error = None
        library, error = self.mtl_device.newLibraryWithSource_options_error_(
            source, compile_options, error
        )
        
        if error:
            raise RuntimeError(f"Failed to compile Metal shader: {error}")
        
        self.mtl_library = library
        
     
        # todo: forward kernel is broken (probably backwards too) 
        forward_function, error = self.mtl_library.newFunctionWithName_("flash_attention_forward")
        if error:
            raise RuntimeError(f"Failed to create forward function: {error}")
            
        self.forward_pipeline, error = self.mtl_device.newComputePipelineStateWithFunction_error_(
            forward_function, error
        )
        
        if error:
            raise RuntimeError(f"Failed to create forward pipeline: {error}")
            
        backward_function, error = self.mtl_library.newFunctionWithName_("flash_attention_backward")
        if error:
            raise RuntimeError(f"Failed to create backward function: {error}")
            
        self.backward_pipeline, error = self.mtl_device.newComputePipelineStateWithFunction_error_(
            backward_function, error
        )
        
        if error:
            raise RuntimeError(f"Failed to create backward pipeline: {error}")
        
        # Create command queue
        self.command_queue = self.mtl_device.newCommandQueue()
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, num_heads, seq_len, head_dim = query.shape
        

        query_buffer = query.contiguous().to(self.device)
        key_buffer = key.contiguous().to(self.device)
        value_buffer = value.contiguous().to(self.device)
        

        output = torch.zeros_like(query)
        output_buffer = output.contiguous().to(self.device)
        

        m = torch.full((batch_size, num_heads, seq_len), float('-inf'), device=self.device)
        l = torch.zeros((batch_size, num_heads, seq_len), device=self.device)
        

        command_buffer = self.command_queue.commandBuffer()
        compute_encoder = command_buffer.computeCommandEncoder()
        

        compute_encoder.setComputePipelineState(self.forward_pipeline)
        

        compute_encoder.setBuffer(query_buffer, offset=0, index=0)
        compute_encoder.setBuffer(key_buffer, offset=0, index=1)
        compute_encoder.setBuffer(value_buffer, offset=0, index=2)
        compute_encoder.setBuffer(output_buffer, offset=0, index=3)
        compute_encoder.setBuffer(m, offset=0, index=4)
        compute_encoder.setBuffer(l, offset=0, index=5)
        

        threadgroup_size = (16, 16, 1)
        grid_size = (
            (seq_len + threadgroup_size[0] - 1) // threadgroup_size[0],
            (num_heads + threadgroup_size[1] - 1) // threadgroup_size[1],
            batch_size
        )
        

        compute_encoder.dispatchThreadgroups(grid_size, threadgroup_size)
        compute_encoder.endEncoding()
        

        command_buffer.commit()
        command_buffer.waitUntilCompleted()
        
        return output
    
    def backward(
        self,
        grad_output: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        m: torch.Tensor,
        l: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, num_heads, seq_len, head_dim = query.shape
        

        grad_output_buffer = grad_output.contiguous().to(self.device)
        query_buffer = query.contiguous().to(self.device)
        key_buffer = key.contiguous().to(self.device)
        value_buffer = value.contiguous().to(self.device)
        m_buffer = m.contiguous().to(self.device)
        l_buffer = l.contiguous().to(self.device)
        

        grad_query = torch.zeros_like(query)
        grad_key = torch.zeros_like(key)
        grad_value = torch.zeros_like(value)
        
        grad_query_buffer = grad_query.contiguous().to(self.device)
        grad_key_buffer = grad_key.contiguous().to(self.device)
        grad_value_buffer = grad_value.contiguous().to(self.device)
        

        command_buffer = self.command_queue.commandBuffer()
        compute_encoder = command_buffer.computeCommandEncoder()
        

        compute_encoder.setComputePipelineState(self.backward_pipeline)
        

        compute_encoder.setBuffer(query_buffer, offset=0, index=0)
        compute_encoder.setBuffer(key_buffer, offset=0, index=1)
        compute_encoder.setBuffer(value_buffer, offset=0, index=2)
        compute_encoder.setBuffer(grad_output_buffer, offset=0, index=3)
        compute_encoder.setBuffer(grad_query_buffer, offset=0, index=4)
        compute_encoder.setBuffer(grad_key_buffer, offset=0, index=5)
        compute_encoder.setBuffer(grad_value_buffer, offset=0, index=6)
        compute_encoder.setBuffer(m_buffer, offset=0, index=7)
        compute_encoder.setBuffer(l_buffer, offset=0, index=8)
        

        threadgroup_size = (16, 16, 1)
        grid_size = (
            (seq_len + threadgroup_size[0] - 1) // threadgroup_size[0],
            (num_heads + threadgroup_size[1] - 1) // threadgroup_size[1],
            batch_size
        )
        

        compute_encoder.dispatchThreadgroups(grid_size, threadgroup_size)
        compute_encoder.endEncoding()
        

        command_buffer.commit()
        command_buffer.waitUntilCompleted()
        
        return grad_query, grad_key, grad_value 

def test_flash_attention(device):
    
    import torch
    import torch.nn.functional as F
    
    # Test parameters
    batch_size = 2
    num_heads = 4
    seq_len = 16
    head_dim = 64
    
  
    q = torch.randn(batch_size, num_heads, seq_len, head_dim).to(device)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim).to(device)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim).to(device)
    
    # native python 
    scale = head_dim ** -0.5
    attn = (q @ k.transpose(-2, -1)) * scale
    attn = F.softmax(attn, dim=-1)
    standard_output = attn @ v
    
    flash_output = MetalFlashAttention(device)(q, k, v)
    
    # Compare outputs
    max_diff = torch.max(torch.abs(standard_output - flash_output))
    mean_diff = torch.mean(torch.abs(standard_output - flash_output))
    
    print(f"Maximum difference: {max_diff.item():.6f}")
    print(f"Mean difference: {mean_diff.item():.6f}")
    
    # Verify the differences are small
    assert max_diff < 1e-5, f"Maximum difference {max_diff} is too large"
    assert mean_diff < 1e-6, f"Mean difference {mean_diff} is too large"
    
    print("Test passed! Flash Attention matches standard attention output.")

if __name__ == "__main__":
    device = torch.device('mps')
    print(f"Using device: {device}")
    test_flash_attention(device) 