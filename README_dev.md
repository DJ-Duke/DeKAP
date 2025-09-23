# DeKAP
Implementation of DeKAP

We are actively preparing to release the code of DeKAP. Please stay tuned!

## Contents
主要程序在process下.
`distillation.py`是有关模型在不同的rank下进行蒸馏的程序。
`allocation.py`是对蒸馏后的模型在网络间进行优化部署的程序。

## Running Steps

### Install Packages

### Setup
调整工作目录到项目根目录下， pip install -e . （开发者模式安装）
如果你之前是用开发模式安装的（pip install -e .），只要你没有改动包的名称或结构，一般不需要重新安装，代码的更改会自动生效。
但如果你改动了依赖、包名、入口等 setup.py 相关内容，建议重新执行

<!---------------------------------------->
# 我自己的测试

