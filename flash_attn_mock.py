# Mock flash_attn module to allow model loading without actual flash attention
# This is a workaround for systems without CUDA or where flash_attn won't compile

class FlashAttention:
    """Mock FlashAttention class"""
    pass

def flash_attn_func(*args, **kwargs):
    """Mock function - not actually used"""
    raise NotImplementedError("Flash attention not available. Model will use fallback attention.")

__all__ = ['FlashAttention', 'flash_attn_func']
