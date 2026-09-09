# @Time    : 2025/5/26 23:16
# @Author  : liuzhou
# @File    : cuda_test.py
# @software: PyCharm
import torch
print(torch.cuda.is_available())  # 应输出True
print(torch.randn(10).cuda())     # 无报错则表示成功调用GPU[<sup data-citation='{&quot;url&quot;:&quot;https://iifx.dev/en/articles/334427007&quot;,&quot;title&quot;:&quot;PyTorch GPU Acceleration: Fixing \&quot;Torch not compiled with CUDA\&quot; Error&quot;,&quot;content&quot;:&quot;torch.cuda.is_available(): This function checks if PyTorch can access a CUDA-enabled GPU. If it returns False, it means CUDA is not properly configured or the PyTorch version lacks CUDA support. torch.randn(10).cuda(): This line creates a random tensor and attempts to move it to the&quot;}'>2</sup>](https://iifx.dev/en/articles/334427007)[<sup data-citation='{&quot;url&quot;:&quot;https://stackoverflow.com/questions/57814535/assertionerror-torch-not-compiled-with-cuda-enabled-in-spite-upgrading-to-cud&quot;,&quot;title&quot;:&quot;\&quot;AssertionError: Torch not compiled with CUDA enabled\&quot; in spite ...&quot;,&quot;content&quot;:&quot;AssertionError: Torch not compiled with CUDA enabled (depite several reinstallations) Hot Network Questions Contracting the First-Person Singular Präteritum&quot;}'>4</sup>](https://stackoverflow.com/questions/57814535/assertionerror-torch-not-compiled-with-cuda-enabled-in-spite-upgrading-to-cud)
