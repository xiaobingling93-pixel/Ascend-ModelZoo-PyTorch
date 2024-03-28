import numpy as np


def save_data_to_npy(data, filename, dtype=None):
    # 创建数组
    if isinstance(data, (int, float)):  # 如果数据是单个数字
        data_array = np.array(data, dtype=dtype)
    elif isinstance(data, tuple) or isinstance(data, list):  # 如果数据是形状元组或列表
        data_array = np.zeros(data, dtype=dtype)
    else:
        raise ValueError("Invalid data type. Data must be a single number or a shape tuple/list.")

    # 保存为 npy 文件
    np.save(filename, data_array)

    print(f"Saved {data} to {filename}")


save_data_to_npy((1, 100, 80), 'x.npy', dtype=np.float32)
save_data_to_npy(100, 'x_lens.npy', dtype=np.int64)
save_data_to_npy((1, 2), 'y.npy', dtype=np.int64)
save_data_to_npy((1, 512), 'encoder_out.npy', dtype=np.float32)
save_data_to_npy((1, 512), 'decoder_out.npy', dtype=np.float32)