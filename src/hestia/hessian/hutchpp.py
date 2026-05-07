import torch
from typing import Optional

class HutchPlusPlusState:
    """
    Manages the Hutch++ algorithm state for a single layer.
    No longer responsible for loops, only for generating vectors and processing HVP results.
    
    [Memory Optimized Version]:
    Core data (S, Q, Omega, Accumulators) all reside on CPU (storage_device).
    Only temporarily move to GPU during computation.
    """
    def __init__(self, param_numel: int, num_sketch: int, num_query: int, device: str):
        self.numel = param_numel
        self.compute_device = device  # Compute device (GPU)
        self.storage_device = 'cpu'   # Storage device (CPU) - Force CPU to save GPU memory
        
        self.num_sketch = min(num_sketch, param_numel)
        self.num_query = num_query
        
        # Results cache
        self.trace_low = 0.0
        self.trace_res = 0.0
        self.final_trace = 0.0
        
        # Intermediate variables (kept on CPU)
        self.S: Optional[torch.Tensor] = None
        self.Y_accum: Optional[torch.Tensor] = None
        self.Q: Optional[torch.Tensor] = None
        self.G_accum: Optional[torch.Tensor] = None
        self.Omega: Optional[torch.Tensor] = None
        self.Z_accum: Optional[torch.Tensor] = None
        
        # Counters
        self.batch_count = 0

    def init_phase1_sketch(self):
        """Phase 1: Generate S (CPU)"""
        # Generate directly on Storage Device
        self.S = torch.randint(0, 2, (self.numel, self.num_sketch), device=self.storage_device).float() * 2 - 1
        self.Y_accum = torch.zeros_like(self.S) # Accumulate H @ S
        self.batch_count = 0
        return self.S

    def accumulate_phase1(self, hvp_chunk: torch.Tensor):
        # hvp_chunk from GPU, move back to Storage Device before accumulating
        self.Y_accum += hvp_chunk.to(self.storage_device)

    def finalize_phase1(self, num_batches: int):
        Y_avg = self.Y_accum / max(1, num_batches)
        # QR decomposition (can be done on CPU, SVD/QR is not CPU-intensive and numerically stable)
        self.Q, _ = torch.linalg.qr(Y_avg, mode='reduced')
        # Cleanup
        self.S = None
        self.Y_accum = None
        
    def init_phase2_subspace(self):
        """Phase 2: Prepare Q (CPU)"""
        self.G_accum = torch.zeros_like(self.Q) 
        self.batch_count = 0
        return self.Q

    def accumulate_phase2(self, hvp_chunk: torch.Tensor):
        self.G_accum += hvp_chunk.to(self.storage_device)
        
    def finalize_phase2(self, num_batches: int):
        G_avg = self.G_accum / max(1, num_batches)
        # Trace calculation
        self.trace_low = torch.sum(self.Q * G_avg).item()
        self.G_accum = None 
        
        if self.num_sketch >= self.numel:
            self.final_trace = self.trace_low
            return True 
        return False

    def init_phase3_residual(self):
        """Phase 3: Generate Omega and project (CPU)"""
        self.Omega = torch.randint(0, 2, (self.numel, self.num_query), device=self.storage_device).float() * 2 - 1
        
        # Matrix multiplication on CPU
        Omega_proj = self.Q @ (self.Q.T @ self.Omega)
        Omega_perp = self.Omega - Omega_proj
        
        self.Z_accum = torch.zeros_like(self.Omega)
        self.batch_count = 0
        return Omega_perp 

    def accumulate_phase3(self, hvp_chunk: torch.Tensor):
        self.Z_accum += hvp_chunk.to(self.storage_device)
        
    def finalize_phase3(self, num_batches: int):
        Y_perp_avg = self.Z_accum / max(1, num_batches)
        
        Y_perp_proj = self.Q @ (self.Q.T @ Y_perp_avg)
        Z = Y_perp_avg - Y_perp_proj
        
        self.trace_res = torch.sum(self.Omega * Z).item() / self.num_query
        self.final_trace = self.trace_low + self.trace_res
        
        self.Q = None
        self.Omega = None
        self.Z_accum = None
