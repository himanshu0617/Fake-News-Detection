"""
Utilities for BERT model optimization and CUDA kernel compilation.
"""

import os
import subprocess
import torch
import torch.nn as nn
from pathlib import Path


class CUDACompiler:
    """Compile and manage CUDA kernels."""
    
    @staticmethod
    def compile_kernels(kernel_file: str = 'attention_kernel.cu', output_dir: str = 'cuda_kernels'):
        """
        Compile CUDA kernels using nvcc.
        
        Args:
            kernel_file: Path to .cu file
            output_dir: Output directory for compiled kernels
        """
        print(f"Compiling CUDA kernels from {kernel_file}...")
        
        # Create output directory
        Path(output_dir).mkdir(exist_ok=True)
        
        # Check if nvcc is available
        try:
            result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
            print(f"CUDA Compiler: {result.stdout.split(chr(10))[0]}")
        except FileNotFoundError:
            print("Warning: CUDA compiler (nvcc) not found in PATH")
            print("Make sure CUDA Toolkit is installed and PATH is configured")
            return False
        
        # Compile to PTX (intermediate representation)
        ptx_file = os.path.join(output_dir, 'attention_kernels.ptx')
        cmd = [
            'nvcc',
            '-ptx',
            '-arch=sm_70',  # For RTX cards, adjust for your GPU
            '-O3',          # Optimization level
            '-use_fast_math',
            kernel_file,
            '-o', ptx_file
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                print(f"✓ Successfully compiled to {ptx_file}")
                return True
            else:
                print(f"✗ Compilation failed:")
                print(result.stderr)
                return False
        except Exception as e:
            print(f"✗ Error during compilation: {e}")
            return False
    
    @staticmethod
    def get_cuda_capability():
        """Get CUDA compute capability of the device."""
        if not torch.cuda.is_available():
            print("CUDA not available")
            return None
        
        device_idx = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device_idx)
        
        print(f"Device: {props.name}")
        print(f"Compute Capability: {props.major}.{props.minor}")
        print(f"Total Memory: {props.total_memory / 1e9:.2f} GB")
        
        return f"sm_{props.major}{props.minor}"


class ModelOptimizer:
    """Optimize BERT model for inference and training."""
    
    @staticmethod
    def quantize_model(model: nn.Module, method: str = 'dynamic') -> nn.Module:
        """
        Quantize model to reduce size and improve inference speed.
        
        Args:
            model: PyTorch model
            method: 'dynamic' or 'static' quantization
        
        Returns:
            Quantized model
        """
        print(f"Quantizing model using {method} quantization...")
        
        if method == 'dynamic':
            quantized_model = torch.quantization.quantize_dynamic(
                model,
                {torch.nn.Linear},
                dtype=torch.qint8
            )
        else:
            # Static quantization requires calibration
            model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
            torch.quantization.prepare(model, inplace=True)
            # Calibration step would go here
            torch.quantization.convert(model, inplace=True)
            quantized_model = model
        
        print("✓ Model quantized successfully")
        return quantized_model
    
    @staticmethod
    def prune_model(model: nn.Module, pruning_ratio: float = 0.1) -> nn.Module:
        """
        Prune model weights to reduce size.
        
        Args:
            model: PyTorch model
            pruning_ratio: Fraction of weights to prune (0.1 = 10%)
        
        Returns:
            Pruned model
        """
        print(f"Pruning model with ratio {pruning_ratio}...")
        
        from torch.nn.utils import prune
        
        for module in model.modules():
            if isinstance(module, torch.nn.Linear):
                prune.l1_unstructured(module, name='weight', amount=pruning_ratio)
                prune.remove(module, 'weight')
        
        print("✓ Model pruned successfully")
        return model
    
    @staticmethod
    def get_model_stats(model: nn.Module) -> dict:
        """
        Get model statistics (parameters, FLOPs, memory).
        
        Args:
            model: PyTorch model
        
        Returns:
            Dictionary with model statistics
        """
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Estimate memory usage (float32 = 4 bytes per parameter)
        model_size_mb = total_params * 4 / (1024**2)
        
        stats = {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': model_size_mb,
            'layers': len(list(model.modules()))
        }
        
        print("\nModel Statistics:")
        print(f"  Total Parameters: {total_params:,}")
        print(f"  Trainable Parameters: {trainable_params:,}")
        print(f"  Model Size: {model_size_mb:.2f} MB")
        print(f"  Layers: {stats['layers']}")
        
        return stats
    
    @staticmethod
    def estimate_flops(batch_size: int = 1, seq_len: int = 512, d_model: int = 768,
                       num_layers: int = 12) -> float:
        """
        Estimate FLOPs for single forward pass.
        
        Args:
            batch_size: Batch size
            seq_len: Sequence length
            d_model: Model dimension
            num_layers: Number of layers
        
        Returns:
            Estimated FLOPs
        """
        # Attention computation: Q@K^T + softmax@V
        attention_flops = 2 * batch_size * num_layers * seq_len * seq_len * d_model
        
        # Feed-forward: Linear1 + ReLU + Linear2
        d_ff = d_model * 4  # typical multiplier
        ff_flops = 2 * batch_size * num_layers * seq_len * d_model * d_ff
        
        # Embedding and output layers
        other_flops = 2 * batch_size * seq_len * d_model * d_model
        
        total_flops = attention_flops + ff_flops + other_flops
        
        print(f"\nFLOPs Estimation:")
        print(f"  Batch Size: {batch_size}")
        print(f"  Sequence Length: {seq_len}")
        print(f"  Model Dim: {d_model}")
        print(f"  Layers: {num_layers}")
        print(f"  Total FLOPs: {total_flops / 1e9:.2f}B")
        print(f"  Estimated Time (100 TFLOPS): {total_flops / 1e12 * 0.01:.3f}s")
        
        return total_flops


class BenchmarkUtil:
    """Benchmark model performance."""
    
    @staticmethod
    def benchmark_inference(model: nn.Module, batch_size: int = 32, seq_len: int = 512,
                          num_iterations: int = 100, device: str = 'cuda') -> dict:
        """
        Benchmark inference speed.
        
        Args:
            model: PyTorch model
            batch_size: Batch size
            seq_len: Sequence length
            num_iterations: Number of iterations
            device: Device to use
        
        Returns:
            Benchmark results
        """
        import time
        
        model.eval()
        model = model.to(device)
        
        # Dummy input
        input_ids = torch.randint(0, 30522, (batch_size, seq_len), device=device)
        token_type_ids = torch.zeros_like(input_ids, device=device)
        attention_mask = torch.ones(batch_size, seq_len, device=device)
        
        # Warmup
        for _ in range(10):
            with torch.no_grad():
                _ = model(input_ids, token_type_ids, attention_mask)
        
        if device == 'cuda':
            torch.cuda.synchronize()
        
        # Benchmark
        start_time = time.time()
        for _ in range(num_iterations):
            with torch.no_grad():
                _ = model(input_ids, token_type_ids, attention_mask)
        
        if device == 'cuda':
            torch.cuda.synchronize()
        
        end_time = time.time()
        total_time = end_time - start_time
        avg_time = total_time / num_iterations
        throughput = batch_size / avg_time
        
        results = {
            'total_time': total_time,
            'avg_time_ms': avg_time * 1000,
            'throughput_samples_per_sec': throughput,
            'batch_size': batch_size,
            'seq_len': seq_len,
            'device': device
        }
        
        print(f"\nInference Benchmark ({device}):")
        print(f"  Batch Size: {batch_size}")
        print(f"  Seq Length: {seq_len}")
        print(f"  Iterations: {num_iterations}")
        print(f"  Avg Time: {avg_time * 1000:.2f} ms")
        print(f"  Throughput: {throughput:.2f} samples/sec")
        
        return results
    
    @staticmethod
    def benchmark_training(model: nn.Module, data_loader, criterion, num_epochs: int = 1,
                          device: str = 'cuda') -> dict:
        """
        Benchmark training speed.
        
        Args:
            model: PyTorch model
            data_loader: Training data loader
            criterion: Loss function
            num_epochs: Number of epochs
            device: Device to use
        
        Returns:
            Benchmark results
        """
        import time
        
        model.train()
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
        
        times = []
        
        for epoch in range(num_epochs):
            start_time = time.time()
            
            for batch_idx, batch in enumerate(data_loader):
                input_ids = batch['input_ids'].to(device)
                token_type_ids = batch['token_type_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)
                
                # Forward
                logits, _ = model(input_ids, token_type_ids, attention_mask)
                loss = criterion(logits, labels)
                
                # Backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            end_time = time.time()
            epoch_time = end_time - start_time
            times.append(epoch_time)
        
        avg_epoch_time = sum(times) / len(times)
        samples_per_sec = len(data_loader.dataset) / avg_epoch_time
        
        results = {
            'avg_epoch_time': avg_epoch_time,
            'samples_per_sec': samples_per_sec,
            'epochs': num_epochs,
            'device': device
        }
        
        print(f"\nTraining Benchmark ({device}):")
        print(f"  Epochs: {num_epochs}")
        print(f"  Avg Epoch Time: {avg_epoch_time:.2f}s")
        print(f"  Throughput: {samples_per_sec:.2f} samples/sec")
        
        return results


if __name__ == '__main__':
    # Example usage
    print("BERT Model Optimization Utilities")
    print("="*60)
    
    # Check CUDA
    compiler = CUDACompiler()
    cuda_cap = compiler.get_cuda_capability()
    
    # Try to compile kernels
    # compiler.compile_kernels()
    
    print("\n" + "="*60)
    print("For more utilities, import this module in your code")
