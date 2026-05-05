完整命令：
# 1. 克隆 v0.8.5 版本
git clone -b v0.8.5 https://github.com/OpenRLHF/OpenRLHF.git

# 2. 重命名为 CRM
mv OpenRLHF CRM

# 3. 进入 CRM 目录（后续 docker -v 会挂载这个目录）
cd CRM

docker run --runtime=nvidia -it --shm-size="10g" --cap-add=SYS_ADMIN \
  --name CRM \
  -v $PWD:/workspace/CRM \
  nvcr.io/nvidia/pytorch:25.02-py3 bash

# 1. 卸载冲突包（安全起见）
pip uninstall xgboost transformer_engine flash_attn pynvml -y

# 2. 设置清华源
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# （不需要）. 安装 v0.8.5 的 openrlhf[vllm]（从 GitHub tag 安装）
pip install "openrlhf[vllm] @ git+https://github.com/OpenRLHF/OpenRLHF.git@v0.8.5"

# 3. 进入挂载目录
cd /workspace/CRM

# 4. 以 editable 模式安装（确保用的是你本地 v0.8.5 代码）
pip install -e .