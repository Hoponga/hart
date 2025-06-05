import torch
import torch.nn as nn
import time
from typing import Dict, List
import numpy as np
from torch.profiler import profile, record_function, ProfilerActivity
import functools
import logging

class Timer:
    def __init__(self, name: str):
        self.name = name
        self.start_time = None
        self.times: List[float] = []
    
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, *args):
        end_time = time.perf_counter()
        self.times.append(end_time - self.start_time)
    
    def get_stats(self) -> Dict[str, float]:
        if not self.times:
            return {"mean": 0, "std": 0, "min": 0, "max": 0}
        return {
            "mean": np.mean(self.times),
            "std": np.std(self.times),
            "min": np.min(self.times),
            "max": np.max(self.times)
        }

class DiTBenchmark:
    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model
        self.device = device
        self.timers: Dict[str, Timer] = {}
        self.memory_stats: Dict[str, List[float]] = {}
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("DiTBenchmark")
    
    def _get_memory_stats(self) -> Dict[str, float]:
        if torch.cuda.is_available():
            return {
                "allocated": torch.cuda.memory_allocated() / 1024**2,  # MB
                "reserved": torch.cuda.memory_reserved() / 1024**2,    # MB
                "max_allocated": torch.cuda.max_memory_allocated() / 1024**2  # MB
            }
        return {}
    
    def benchmark_forward(self, x: torch.Tensor, t: torch.Tensor, num_runs: int = 100):
        """Benchmark the forward pass of the model."""
        self.logger.info("Starting forward pass benchmarking...")
        
        # Warmup
        for _ in range(10):
            with torch.no_grad():
                _ = self.model(x, t)
        
        # Main benchmarking
        for i in range(num_runs):
            self.logger.info(f"Run {i+1}/{num_runs}")
            
            # Clear cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Record memory before
            mem_before = self._get_memory_stats()
            
            # Forward pass with profiling
            with torch.no_grad():
                with profile(
                    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                    record_shapes=True,
                    profile_memory=True,
                    with_stack=True
                ) as prof:
                    with record_function("model_forward"):
                        output = self.model(x, t)
            
            # Record memory after
            mem_after = self._get_memory_stats()
            
            # Print profiling results
            self.logger.info("\nProfiling Results:")
            self.logger.info(prof.key_averages().table(
                sort_by="cuda_time_total", row_limit=10))
            
            # Record memory stats
            for key in mem_after:
                if key not in self.memory_stats:
                    self.memory_stats[key] = []
                self.memory_stats[key].append(mem_after[key] - mem_before[key])
    
    def benchmark_attention(self, x: torch.Tensor, num_runs: int = 100):
        """Benchmark the attention mechanism specifically."""
        self.logger.info("Starting attention benchmarking...")
        
        # Get attention module
        attention_module = None
        for module in self.model.modules():
            if hasattr(module, 'qkv'):
                attention_module = module
                break
        
        if attention_module is None:
            self.logger.error("Could not find attention module in the model")
            return
        
        # Warmup
        for _ in range(10):
            with torch.no_grad():
                _ = attention_module(x)
        
        # Main benchmarking
        for i in range(num_runs):
            self.logger.info(f"Run {i+1}/{num_runs}")
            
            with torch.no_grad():
                with profile(
                    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                    record_shapes=True,
                    profile_memory=True,
                    with_stack=True
                ) as prof:
                    with record_function("attention_forward"):
                        _ = attention_module(x)
            
            self.logger.info("\nAttention Profiling Results:")
            self.logger.info(prof.key_averages().table(
                sort_by="cuda_time_total", row_limit=10))
    
    def benchmark_memory_usage(self, x: torch.Tensor, t: torch.Tensor, num_runs: int = 10):
        """Benchmark memory usage during forward and backward passes."""
        self.logger.info("Starting memory usage benchmarking...")
        
        for i in range(num_runs):
            self.logger.info(f"Run {i+1}/{num_runs}")
            
            # Clear cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Forward pass
            output = self.model(x, t)
            
            # Backward pass
            loss = output.sum()
            loss.backward()
            
            # Record memory stats
            mem_stats = self._get_memory_stats()
            for key, value in mem_stats.items():
                if key not in self.memory_stats:
                    self.memory_stats[key] = []
                self.memory_stats[key].append(value)
            
            # Clear gradients
            self.model.zero_grad()
    
    def print_results(self):
        """Print benchmarking results."""
        self.logger.info("\n=== Benchmarking Results ===")
        
        # Print memory statistics
        self.logger.info("\nMemory Statistics (MB):")
        for key, values in self.memory_stats.items():
            if values:
                self.logger.info(f"{key}:")
                self.logger.info(f"  Mean: {np.mean(values):.2f}")
                self.logger.info(f"  Max: {np.max(values):.2f}")
                self.logger.info(f"  Min: {np.min(values):.2f}")

def benchmark_dit_model(model: nn.Module, batch_size: int = 1, image_size: int = 32, num_runs: int = 100):
    """Main benchmarking function."""
    device = next(model.parameters()).device
    
    # Create sample input
    x = torch.randn(batch_size, 3, image_size, image_size, device=device)
    t = torch.randint(0, 1000, (batch_size,), device=device)
    
    # Create benchmarker
    benchmarker = DiTBenchmark(model, device)
    
    # Run benchmarks
    benchmarker.benchmark_forward(x, t, num_runs)
    benchmarker.benchmark_attention(x, num_runs)
    benchmarker.benchmark_memory_usage(x, t, num_runs=10)
    
    # Print results
    benchmarker.print_results()

if __name__ == "__main__":
    # Example usage
    from dit import DiT  # Import your DiT model
    
    # Create model
    model = DiT(
        input_size=32,
        patch_size=4,
        in_channels=3,
        hidden_size=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        num_classes=1000,
        learn_sigma=True,
    ).cuda()
    
    # Run benchmarks
    benchmark_dit_model(model) 