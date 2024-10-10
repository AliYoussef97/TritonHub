import torch
import torch.nn as nn
from softmax import Softmax
from tabulate import tabulate as tb

class SoftmaxUnitTest:

    def __init__(self, B=4, N=512, M=512, D=256, image=True, dtype=torch.float32, print_tb=False):
        self.B = B
        self.N = N
        self.M = M
        self.D = D
        self.image = image
        self.dtype = dtype
        self.print_tb = print_tb

        # Initialize the softmax functions
        self.softmax = Softmax
        self.softmax_torch = nn.Softmax(dim=-1)

    def run(self):
        # Create the input tensor with the specified shape and initialize it
        if self.image:
            input_data = torch.randn(self.B, self.M, self.N, self.D, device='cuda', dtype=self.dtype)
        else:
            input_data = torch.randn(self.B, self.M * self.N, self.D, device='cuda', dtype=self.dtype)

        # Create separate tensors for input and input_ref using the same data and ensure gradient computation
        input = input_data.clone().detach().requires_grad_(True)
        input_ref = input_data.clone().detach().requires_grad_(True)

        # Set the tolerance for the comparison
        rtol, atol = rtol, atol = (3e-4, 1e-3) if dtype == torch.float32 else (5e-3, 1e-2)
        if dtype == torch.float16:
            rtol, atol = 1e-2, 5e-2
        self.forward(input, input_ref, atol, rtol)

    def forward(self, input, input_ref, atol, rtol):
    
        output = self.softmax(input)  # This run measures the compiled time
        output_ref = self.softmax_torch(input_ref)
        assert torch.allclose(output, output_ref, atol=atol, rtol=rtol), 'Error in forward pass'
        if self.print_tb:
            diff = (output - output_ref).abs()
            print(tb([[diff.mean().item(), diff.max().item()]],
                    headers=['Forward Mean difference', 'Forward Max difference'], tablefmt='orgtbl'))
        self.backward(input, input_ref, output, output_ref, atol, rtol)

    def backward(self, input, input_ref, output, output_ref, atol, rtol):
        # Define a simple loss as the sum of softmax outputs and perform backward pass
        loss = output.sum()
        loss_ref = output_ref.sum()

        loss.backward()
        loss_ref.backward()

        # Check backward pass results
        assert torch.allclose(input.grad, input_ref.grad, atol=atol, rtol=rtol), 'Error in backward pass'
        if self.print_tb:
            diff = (input.grad - input_ref.grad).abs()
            print(tb([[diff.mean().item(), diff.max().item()]], 
                    headers=['Backward Mean difference', 'Backward Max difference'], tablefmt='orgtbl'))

if __name__ == '__main__':
    B, N, M = 1, 256, 256
    image = True
    print_tb = False
    for i in range(2):
        if i ==0: print('First iteration Slow due to Triton Autotune')
        for D in [32, 64, 128, 256, 512, 1024, 2048]:
            for dtype in [torch.float16, torch.float32, torch.float64]:
                runner = SoftmaxUnitTest(B, N, M, D, image, dtype, print_tb)
                runner.run()
                print(f'All tests passed at Dimensions: {D}, dtype: {dtype}')
    print('All tests passed!')