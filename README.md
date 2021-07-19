
# SoftWare Cup
2021年中国软件杯A2组赛题，基于百度飞桨的多目标行人追踪系统
# 基于百度飞桨的单/多镜头多目标行人追踪系统  

##  简介  

***

我们的多目标行人追踪系统基于百度飞桨的PaddleDetection框架进行模型的训练，并使用PyCharm进行软件的开发。  

我们的前端UI设计采用的是PyQt5进行开发设计，图像处理的主要框架采用的是百度飞桨的PaddleDetection进行部署开发，对整个项目使用pyinstaller进行打包封装，并生成了可脱机执行的exe文件。

我们的系统经过测试，可以满足在多个场景下进行行人识别追踪，对于百度官方发布的数据集，我们的模型识别精度高达***72.542%***。

***

##  运行环境说明 

我们的系统目前只适用于windows操作系统，并且当前只适用于windows10版本。

| 环境配置          | 版本   |
| ----------------- | ------ |
| paddle paddle-gpu | 2.1.0  |
| python            | 3.7.10 |
| CUDA              | 10.2   |
| cudnn             | 7.6.5  |

## 相关依赖包说明  

[相关库的yaml文件](https://github.com/JunLei01/SoftWare/blob/master/environment.yaml)  *你在创建conda环境时可以快速安装项目所需包的 yaml文件*

[相关库的txt文件](https://github.com/JunLei01/SoftWare/blob/master/requirement.txt)  *你在创建conda环境时可以快速安装项目所需包的 txt文件*

