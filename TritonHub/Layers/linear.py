import torch
import triton
import triton.language as tl
import math

def get_cuda_autotune_config():
    return [
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1}, num_stages=3,
                      num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 1}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 1}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 1}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 1}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 1}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 1}, num_stages=5,
                      num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 1}, num_stages=5,
                      num_warps=2),
        # Good config for fp8 inputs.
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=3,
                      num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=3,
                      num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4)
    ]

@triton.autotune(
    configs=get_cuda_autotune_config(),
    key=['M', 'N', 'K'],
)
@triton.jit
def _linear_kernel_fwd(X,
                       W,
                       Y,
                       M, 
                       N,
                       K,
                       stride_am, stride_ak,
                       stride_bk, stride_bn,
                       stride_cm, stride_cn,
                       B,
                       BLOCK_SIZE_M: tl.constexpr,
                       BLOCK_SIZE_N: tl.constexpr,
                       BLOCK_SIZE_K: tl.constexpr,
                       GROUP_SIZE_M: tl.constexpr,
                       HAS_BIAS: tl.constexpr):
        
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    X = X + (offs_am[:, None] * stride_am  + offs_k[None, :] * stride_ak)
    W = W + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    y = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        x = tl.load(X, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        w = tl.load(W, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        y = tl.dot(x, w, y)
        X += BLOCK_SIZE_K * stride_ak
        W += BLOCK_SIZE_K * stride_bk
    if HAS_BIAS:
        y += tl.load(B + offs_bn, mask=offs_bn < N)
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    Y = Y + stride_cm  * offs_cm[:, None] + stride_cn  * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(Y, y, mask=c_mask)
    
def _linear_fwd(x, weight, bias):
    if x.stride(-1) != 1:
        x = x.contiguous()
    batch_shape = x.shape[:-1]
    x = x.reshape(-1, x.shape[-1])
    assert x.stride(-1) == 1, 'expect input to be row-major'
    M, K = x.shape
    N = weight.shape[-1]
    weight = weight.contiguous()
    assert weight.stride(-1) == 1, 'expect weight to be row-major'
    out = torch.empty((M, N), dtype=x.dtype, device=x.device)
    assert out.stride(-1) == 1, 'expect output to be row-major'
    HAS_BIAS = True if bias is not None else False
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )
    with torch.cuda.device(x.device.index):
        _linear_kernel_fwd[grid](x,
                                 weight,
                                 out,
                                 M,
                                 N,
                                 K,
                                 x.stride(0), x.stride(1),
                                 weight.stride(0), weight.stride(1),
                                 out.stride(0), out.stride(1),
                                 bias,
                                 HAS_BIAS=HAS_BIAS)                  
    return out.reshape(*batch_shape, N)

# @triton.autotune(
#     configs=get_cuda_autotune_config(),
#     key=['M', 'N', 'K'],
# )
# @triton.jit
# def _linear_kernel_bwd():
#     # TODO: Implement the backward pass.
#     pass
    

def _linear_bwd(x, dout, weight, bias):
    # TODO: Implement the backward pass.
    raise NotImplementedError("Backward pass for linear layer is not implemented yet.")
 
class linear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias):
        output = _linear_fwd(input, weight, bias)
        ctx.save_for_backward(input, weight, bias)
        return output

    @staticmethod
    def backward(ctx, d_out):
        input, weight, bias = ctx.saved_tensors
        grad, dw, db = _linear_bwd(input, d_out, weight, bias)
        return grad, dw, db

class Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(torch.empty(in_features, out_features, **factory_kwargs))
        if bias:
            self.bias = torch.nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self.linear_fn = linear.apply
    
    def reset_parameters(self) -> None:
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        return self.linear_fn(x, self.weight, self.bias)