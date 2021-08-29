##实测环境配置：
Windows10/Ubuntu18.04
python3.8
cuda10.2
cudnn7.6.5（不支持8.0以上）

- 所需python库函数:
paddlepaddle-gpu
paddleocr==2.0.6
以及requirements.txt中所需的其他依赖。

- 安装时建议采用百度镜像，如：
‘’‘
pip install paddlepaddle-gpu -i https://mirror.baidu.com/pypi/simple
’‘’

## 快速开始
在安装完所需依赖及环境配置后，用测试视频快速测试：
‘’‘
python demo_0828.py
’‘’
- 测试视频位置为：./test_data/video/
- 输出结果位置为:./inference_results/jsons/
- 模型位置为:./inference/

## 待解决的问题
- 抽帧目前使用ffmpeg，与先前的其他模块的opencv抽帧不同。逻辑为检测所有帧——提取关键结果。
- 若其他算法模块均采用关键帧提取——再检测的方法，可修改抽帧方法以便能够统一视频前处理，提升性能。
- 新算法自带结果可视化的功能，目前默认保存在./inference_results目录下，届时前端需要可视化结果的话可做进一步开发。

## 关于代码
demo_0828.py内直接看main函数，三个输入的变量很好识别。
json输出在203行左右，可视化结果在后面一点点。没有写成函数参数，要调整的话也很简单。