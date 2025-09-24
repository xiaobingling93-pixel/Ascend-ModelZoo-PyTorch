#!/bin/bash

# 查询NPU设备信息并获取第四列的PCI地址
pci_addresses=$(npu-smi info | grep 0000 | awk '{print $4}')

if [ -z "$pci_addresses" ]; then
    echo "未找到匹配的NPU设备信息"
    exit 1
fi

echo "找到以下NPU设备PCI地址:"
echo "$pci_addresses"

# 为每个PCI地址查询NUMA节点信息
i=0
for  addr in $pci_addresses; do
    echo "查询PCI地址 $addr 的NUMA信息"

    # 使用lspci获取节点信息
    node_info=$(lspci -vvv -s $addr | grep node | awk -F ':' '{print $2}' | awk '{print $1}')
    numa_info=$(lscpu | grep "node${node_info}")
    echo ">>>>设备 $i 对应 NUMA 节点: $node_info, $numa_info"
    i=$((i+1))
done