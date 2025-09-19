1. 导入NPU相关库

```python
import torch
import torch_npu
```

2. 指定NPU作为运行设备
* .to(device)方式

    迁移前：

    ```python
    device = torch.device('cuda:{})'.format(local_rank))
    model.to(device)
    data.to(device)
    ```

    迁移后：

    ```python
    device = torch.device('npu:{}'.format(local_rank))
    model.to(device)
    data.to(device)
    ```

* set_device方式

    迁移前：

    ```python
    torch.cuda.set_device(local_rank)
    ```

    迁移后：

    ```python
    torch_npu.npu.set_device(local_rank)
    ```

    更多torch_npu迁移指导请参考：[PyTorch 模型手工迁移指南](https://www.hiascend.com/document/detail/zh/Pytorch/60RC1/ptmoddevg/trainingmigrguide/PT_LMTMOG_0018.html)

3. 运行模型，确保迁移后可以正确运行。