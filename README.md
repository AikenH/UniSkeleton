# ReadMe

@AikenHong2022 

面向类别长尾分布和增量学习的图像分类框架，该架构中包含自监督预训练、模型训练、增量学习三大模块，可以根据该框架设计通常的图像分类框架。

**TBD**:

- [ ] 补完Config的说明和各种参数的讲解，主要包含一些可选的模式和说明等
- [ ] 编写Custom的指南
- [ ] 修改basic_config等文件，删除Dropout的参数

## Quickstart

首先说明如何使用默认的设置运行**长尾情景**下的图像分类模型（对比-校正模型）

```shell
zsh run_dis.sh
# or
python main.py --cfg config/dis_imb_config.yaml
```

默认设置未启用自监督预训练过程，也未启用后续的增量学习流程，使用Res50作为Backbone，在Cifar100上进行，长尾参数设置为0.1。

建议自定义的基础设置如下：

```yaml

val_n: 1                                                    # 多少个epoch进行一次验证
cuda_devices: "0"                                           # ...
save_pth: save_model                                        # 模型保存路径
resume: false                                               # 是否读取模型恢复训练
training_mode: imb                                          # 使用何种训练方式进行训练，包括['imb','']

logger:
  logname: train_cifar100_resnet50                          # 保存的.log 文件的文件名
  predix: null                                              # 当前训练的子文件夹名，null的话默认使用时间戳
  log_dir: ./log                                            # .log和tensorboard文件的存储路径
  subfix: A_                                                # 当前训练的外层目录名

data:
  prepocess: imb                                            # 数据采样模式选择（imb采样长尾、采样新类、采样长尾后选择新类）

training_opt:
  loss: CE                                                  # 可参考Loss文件夹中自定义
  notzeroloss: 0.12                                         # 损失添加非零偏置项
  optimizer: SGD                                            # 优化器选择，按需自定义
  optim_opt: { lr:0.2 }                                     # 特征提取器lr设置
  clf_optim_opt: { lr: 0.2, momentum: 0.9, weight: 0.0005 } # 默认lr设置，记得随着优化器改变
  train:
    num_epochs: 180,                                        # 训练的epoch
    patience: 200,                                          # Early Stop的容忍上限
  dataopt:
    batch_size: 256
  max_mix_epoch: 180                                        # 在多少个epoch使用mixup
	
```

基于需求定义好配置文件后，修改对应的 `--cfg xxx` 或者在 `run_xxx.sh` 中修改配置文件的位置即可。

## Locate Related Files

按照文件对应的功能确定其文件夹位置，该部分仅展示主要的文件目录，各文件详细实现的功能请查阅[Files Manual](#FManual)

**UniSkeleton**  

├─ **config**: 存放配置文件和命令行解析代码，控制训练任务和模型组成。  
├─ **data**: 数据集创建、采样、增强代码和脚本，以及Collate_Fn等和数据处理有关的类与函数。  
├─ **layers**: 功能单元，例如activation、cluster、Confidence等。（部分和Backbone直接相关的单元可能直接在Backbone中编写）  
├─ **loss**: 顾名思义，损失设计，通过\_\_init\_\_管理损失。  
├─ **model**: Backbone，其中\_\_init\_\_文件编写函数用于组装完整模型。  
├─ **run.sh**: 运行脚本，主要利用脚本指定载入的YAML进行控制训练任务。  
└─ **util**: 其他运行单元，主要包括：Wrap（for logs & tensorboard）、可视化代码、Metric、**运行脚本**、以及其他辅助函数。

需要注意的是，为了在Config中配置模型组成和损失组成，在Loss和Model中的`init.py`，编写了相应代码进行注册，因此实现新的模块后需要仿照其样式进行register。

##  Usage of Modules

该部分介绍如何使用一些模块，以及部分模块的运行逻辑。

### Log_Wrapper

使用python中的**Decorator**机制，通过装饰器修饰函数，便于记录算运行过程，并将相应的数据上传至Tensorboard，避免该类代码重复使用时需要重复编写，该部分代码主要实现于`util-wraps.py`。

- `log_timethis`：记录装饰函数的运行时间，并通过指定的logger进行输出。
- `timethis`：同样记录函数运行时间，但仅使用print输出，不通过logger记录。

> 注意事项（缺陷）：log版本的需要在`util-runningscript.py-init_log`中进行logger的指定，由于第一次编写装饰器，这样的组织并不理想，应使用别的组织方式来进行log的注册

- **BASE** `record_nbatch`：Log和Tensorboard的核心记录装饰器，需要对其初始化writer和logger，该装饰器接受函数的输出（Dict）。
  - Tensorboard部分，通过字典的关键字划分为`loss`、`acc`、`weights`三类，记录相应的参数曲线。
  - Log部分，通过**EPOCH**关键字，对log进行info和debug不同level的输出和记录。
- **TEXT**`text_in_tensorboard`：  Tensorboard的文本装饰器，需要初始化writer，该装饰器的输出接受（Str），使用设定的Title参数，将文本加入Tensorboard。
- **FIGURE**`visual_1s`：Tensorboard的图像装饰器，需要初始化weiter，该装饰器基于PhaseName执行add_figure操作。

该部分代码的介绍大体如上所述，初始化的方式参考`runningscript.py-init_log`，使用的方式如下：

```python
@log_timethis(desc = "training")
def func(*args,**kwargs):
    pass

@visual_1s(phaseName = "first_batches")
def datavisualization(...):
    pass
@visual_1s(phaseName = "tsne result")
def tsne_visual(...):
    pass

@record_nbatch(verbose=False)
def update(*args, **kwargs):
    pass
# 实际使用中可能会涉及到phase的转变如下，
# 其中set_phase是wrapper的函数，
# 具体的使用参考runningScript中的train或test函数
update.set_phase(phase + 'test_epoch')
...
```

## Configuration

配置读取过程包含在argparser中，流程如下：读取命令行参数保存至Args ➡ 读取YAML文件至Configs ➡ 利用Args更新并合并到Configs.

![image-20220523183444443](https://picture-bed-001-1310572365.cos.ap-guangzhou.myqcloud.com/imgs/image-20220523183444443.png)

在完成配置信息的合并后（参考update、getValue），在后续通过传递Configs（Dict）来控制代码的整体运行。

### ArgumentParser

Args用于指定函数运行的Config文件位置，并可指定部分常改的参数，若与YAML中的定义重复，则使用Args中的对其进行覆写。

Args中主要的参数如下：

```yaml
--cfg:            # path of configuration
--val_n:          # ...
--cuda_devices: 1 # cuda version
--resume:         # ToF
--ckpt_pth:       # model files location
--batch_size:     # ...
...
```

如果没有指定cfg的话，直接返回args的参数，但是目前仅凭args的参数无法支持后续训练（建议直接改成报错）

### LoadYAML

由于配置文件较长，因此分别对该配置文件进行讲解，其中，必须包含的模块有： 、而非必须包含的模块有： 、

#### Basic&Log基础配置

该部分包括：设置基础运行环境、Log文件存储位置、Resume的模型、评估的指标、训练模式。  
具体如下：

```yaml
val_n: 1                                       # 多少个epoch进行一次验证
apex: false                                    # 丢弃功能
cuda_devices: "0"                              # Cuda设备设置

resume: true                                   # 恢复训练/载入模型
save_pth: save_model                           # 模型保存目录
ckpt_pth: save_model/ckpt/.../both_scl_7213.pt # 读取的模型地址
metrics_type: !!python/tuple ["acc1", "acc5"]  # 评价指标，可以选择计算的类型，在util-metric.py-Metric_cal中实现

training_mode: imb                             # 训练模式【imb，null，】
logger:
  logname: train_cifar100_resnet50             # .log的文件名
  console_level: 2                             # 设置命令行输出【info，warning，...】
  prefix: null                                 # 当前训练的子文件夹名，null的话默认使用时间戳
  log_dir: ./log                               # .log和tensorboard文件的存储路径
  subfix: final_der_                           # 当前训练的外层目录名
```



#### Model模型配置

通过该部分配置文件来控制模型的Assemble，组合分类器和特征提取器，设定基础的维度等。  
具体如下：

```yaml
networks:
  {
    feat_model: cifar_rs50, # Backbone 模型设置
    cls_model: causal,      # Classifier 设置
    num_cls: 100,           # 最终输出的类别预测（for fc）
    hidden_layers: null,    # 隐含层
    pretrain: false,        # 已丢弃？
    in_dim: 2048,           # 输入Clf的维度（即特征输出的维度）
    feature_vis: true,      # 是否输出特征提取器的提取结果
    activation: null,       # 设置激活函数类型（通常使用模型中自带的激活函数）
    dropout: 0.5,           # ...
  }
```
若要使用自己定义的feat_model（Backbone）和cls_model（Classifier），在编写完相应的模块后，在`model/__init__.py`中注册并赋予其对应的关键词，并完善Assemble函数的相关逻辑。

#### Data数据配置

通过该部分的参数设置，在所有**训练进程之前**初始化各类数据集，可以在后续的训练进程中调用

```yaml
# istrain: 0 train | 1 test  | else val
# imb_factor: 0.1->10 0.02->50, 0.01->100
# strategy: exp
# 上述两者结合起来是长尾的默认实验场景
data:
  preprocess: both                # 数据采样模式选择（imb采样长尾、采样新类、采样长尾后选择新类）
  transformer: cifar100           # dataset使用的数据增强策略
  dataset: {
      datatype: cifar100,         # 使用什么数据集
      save_path: ../DownloadData, # 数据集的存储地址
      istrain: 0,                 # 模式选择，
      needMap: false,             # 需要Check，是否建立标签到Index的映射，
      num_new: 20,                # 如果启用New采样方法，选择多少类别作为新类
      imb_factor: 0.1,            # 如果启用Imb采样，长尾采样参数的设置
      strategy: exp,              # 使用何种长尾采样方法，默认exp，可自己实现其余策略
      decay: 0.5,                 # 已抛弃，用于其余采样方法中的参数衰减
      sort: false,                # 是否对数据集进行排序
      num_cls: 100,               # CHeck
      random_seed: 888,           # ...
    }

```

#### Pretrain预训练控制

```yaml
pretrain:
  ispretrain: false                                                        # 预训练开关
  using_projector: true                                                    #
  diff_lr: false                                                           # 默认False，在该部分暂时没有使用区别的学习率
  pretrain_opt:
    {
      type: train,                                                         # 【Train, Load】 选择使用训练好的预训练模型还是加载预训练模型
      pth: save_model/pre_train/causal_cifar_rs50_cifar100/1024_81_imb.pt, # 模型的存储地址
    }
  epoch: 500                                                               # 预训练的Epoch上限
  network:                                                                 # 模型设计
    {
      feat_model: cifar_rs50,                                              # 特征提取器模型
      cls_model: causal,                                                   # 分类器模型
      num_cls: 10,                                                         # 最终输出维度（projector的输出维度）
      hidden_layers: !!python/tuple [512, 512],                            # fc的隐含层参数
      pretrain: false,                                                     # 丢弃
      in_dim: 2048,                                                        # 特征输出维度，Projector的输入维度
      feature_vis: true,                                                   # 是否输出中间特征（不影响代码运行，可打开监控）
      activation: null,                                                    # 激活函数设置
    }
  datasample: all                                                          # 处理数据用于模型预训练，无标注数据 【all：使用所有数据，imb使用长尾采样处理过的数据】
  dataset:                                                                 # 类似上述的数据设置（主要为了All）
    {
      datatype: cifar10,
      save_path: ../DownloadData,
      istrain: 0,
      num_cls: 10,
      random_seed: 888,
    }
  dataload_opt:                                                            # Dataloader的参数，其实直接对照表就行了
    {
      batch_size: 1024,                                                    # ...
      shuffle: true,                                                       # ...
      num_workers: 28,                                                     # ...
      pin_memory: false,                                                   # ...
      drop_last: true,                                                     # ...
      prefetch_factor: 5,                                                  # ...
    }
  loss_opt: { LOSSTYPE: ntxent, memory_size: 0, temperature: 0.07 }        # 自监督学习损失函数设置
  optim: SGD                                                               # 优化器选择
  optim_opt: { lr: 0.01, momentum: 0.9, weight_decay: 0.0005 }             # 和优化器对应的参数设置
  sche_opt:                                                                # 默认设置为自动学习率衰减模式，自动调节学习率，需要别的策略可以自己定制一下
    {
      mode: max,
      factor: 0.2,
      patience: 10,
      min_lr: !!float 1e-5,
      verbose: true,
    }
```

#### Training训练过程控制

基本的参数控制模块，预训练，增量训练也参照该参数设置的样式进行，作为最主要的参考。

```yaml
training_opt:                                                                            # 基础训练部分参数
  loss: CE                                                                               # 设置基础损失函数类型
  notzeroloss: 0.12                                                                      # 非零损失偏差
  optimizer: SGD                                                                         # 优化器
  optim_opt: { lr: 0.2 }                                                                 # 分类器的学习率设置
  clf_optim_opt: { lr: 0.2, momentum: 0.9, weight_decay: 0.0005 }                        # 判别器和默认的学习率设置
  schedule: multistep                                                                    # 梯度下降方法设置，和下面的sche_opt相互组合
  sche_opt: { milestones: !!python/tuple [60, 120, 160, 190], gamma: 0.1, verbose: true}
  warmup: true                                                                           # 是否启用warmup，warm_up作为其参数设置配合使用
  warm_up: { multiplier: 1, total_epoch: 10 }
  freq_out: 20                                                                           # 多少个batch输出一次log（命令行中）
  train:                                                                                 # 训练参数
    {
      num_epochs: 180,                                                                   # 训练上限
      drop_rate: 0.5,                                                                    # ...
      patience: 200,                                                                     # ES（Early-Stop）范围，是否开启考虑使用的学习率衰减策略
      iterations: 100,                                                                   # 丢弃，暂未启用
      verbose: true,                                                                     # ...
    }
  dataopt:                                                                               # dataloader的参数设置
    {
      batch_size: 256,
      shuffle: true,
      sampler: null,
      num_workers: 28,
      collate_fn: null,
      pin_memory: false,
      drop_last: true,
      prefetch_factor: 5,
    }
  combiner_opt: {strategy: 'linear'}                                                     # 损失结合器，用于选择组合损失的策略
  max_mix_epoch: 180                                                                     # 在多少个epoch中启用mixup策略
  criterion: { alpha: 0, beta: 0 }                                                       # TODO: check this one
```

#### Online在线学习过程控制

该模块主要控制代码的增量学习进程（其中可包含新类发现和深度聚类过程），与Training过程类似的就不再赘述。

```yaml
online:
  enable: true                                                               # 是否启用该进程
  resume: false                                                              # 是否继续未完成的训练，ckpt存储模型地址
  ckpt: save_model/distill/mlp_cifar_rs50_cifar100/2022-01-17_12-07-37_PM.pt
  cluster_t: kmeans                                                          # 用于深度聚类策略中，选择聚类方法
  cluster_dict: { n_clusters: 20, init: k-means++, random_state: 0 }         # 聚类的选项，聚簇数目和初始化方法
  downsample_t: pca                                                          # 选择降维的方法
  downsample_dict: { n_components: 20, random_state: 0 }                     # 降维的选项，最终维度选择
  projecter: false                                                           # TODO:: 检查一下，是否从自监督学习中继承projector
  optimizer: SGD
  optim_opt: { lr: 0.1 }
  clf_optim_opt: { lr: 0.1, momentum: 0.9, weight_decay: 0.0005 }
  notzeroloss: 0.03
  dis_criterion: kd_c                                                        # 选择使用的蒸馏损失类型
  dis_loss_opt: {num_classes: 100, factor: 0.9, temperature: 2.0}            # 蒸馏损失参数，其中最重要的是factor，选择蒸馏权重。
  schedule: multistep
  sche_opt: {
    milestones: !!python/tuple [60,120,180],
    gamma: 0.1,
    verbose: true
    }
  warmup: true
  warmup_opt: { multiplier: 1, total_epoch: 10 }
  combiner_opt: {strategy: 'new_linear'}
  epoches: 220
  mix_epoches: 200
```

## Customization 

### Metric Design 

模型的评价指标如ACC，PR等统计指标的设计和编写，

### Loss  Design

对于损失函数的处理方式：将所有自定义的损失函数 存放在 ./loss中，将torch等已经被集成的常见loss function，放在其中的__init__.py中，在 __inti__中编写L_select，进行import管理，通过loss_t动态的返回相应的loss function，具体的操作如下：

```python
from loss import L_select as loss_setter
loss = loss_setter(loss_type)
```

所以我们就i需要对每个写好的loss function在 `__init__`中进行统一的管理

### Model （Layer）Design

对于backbone的处理方式和Loss是大体一致的）

### recorder & Evaluator

#### recorder

* 按照当前传入的dict 来更新所有评估指标
* 分为两部分输出，一是累加值一是当前值

#### Evaluator


* 编写call function 对采用的metric计算进行管理 方便return和调用

## FManual

详细介绍各个文件实现的功能，便于代码查阅和修改。其中...或无描述，则参见文件名，不在赘述。

```txt
├─ README.md                  # ...
├─ config                     # 配置文件夹
│  ├─ argparser.py            # 解析命令行代码、读取YAML并进行合并更新Config
│  ├─ ...                     # 部分配置文件
├─ data
│  ├─ __init__.py
│  ├─ collators.py            # 定义collate_fn（未启用）
│  ├─ dataAugment.py          # 定义albumentations数据增强类、高斯模糊、对比学习的Duplicate
│  ├─ dataUtils.py            # 读取、迁移、预处理本地数据、部分可视化代码
│  ├─ datasetGe.py            # 数据下载、采样、选择新类、创建&混合Dataset
│  ├─ setTransformer.py       # 预设、选择训练过程中使用的Transformer
│  └─ unzipImageNet.sh        # 批量解压数据集压缩包，将其解压到对应的文件夹中，在解压后删除压缩包
├─ layers
│  ├─ CausalNormClassifier.py
│  ├─ __init__.py
│  ├─ activations.py          # 部分激活函数
│  ├─ classifier.py           # 一些分类器定义（包括MLP等）
│  ├─ clusters.py             # 深度聚类中的一些核心类：聚类方法的生成和调用、伪标签生成和前后对齐
│  ├─ convolution.py          # 分组卷积
│  ├─ ezConfidence.py         # 置信度分析策略
│  └─ warmup.py               # 较为通用的Warmup方法
├─ loss
│  ├─ KD_loss.py              # 蒸馏损失（类和函数两种形式，推荐使用类）
│  ├─ angular_loss.py         # 参考人脸比对中的分类方法，使用角度方法修正softmax（未启用）
│  ├─ NTXent_loss.py          # 基础对比损失
│  ├─ custom_loss.py          # Combiner（损失重加权策略）、KL、Mixup
│  ├─ evidential_loss.py      # 证据分析的损失（未启用）
│  ├─ longtailed_loss.py      # 一些为长尾设置的损失如Disalign（未启用）
│  ├─ memory_bank.py          # MoCo的动量机制
│  └─ supContrast_loss.py     # 监督对比损失
├─ main.py                    # 主入口函数
├─ model
│  ├─ EfficientNet.py         # ...
│  ├─ ResNet.py               # ...
│  ├─ ResNet32.py             # ...
│  ├─ ViTs.py                 # SwinTransformer， Cifar上无法体现其效果，需要进一步验证
│  └─ small_ResNet.py         # 修正resnet以适应Cifar的低分辨率的版本
├─ run.sh                     # 运行脚本
├─ run_dis.sh                 # 同上
├─ util
│  ├─ Visualize.py            # 数据可视化、TSNE可视化等
│  ├─ dis_runner.py           # 对比-蒸馏的核心训练过程、其余训练的流程控制也集成到其中，可启用多种训练方式
|  ├─ lt_runner.py            # 对比-校正的核心训练过程
|  ├─ metric.py               # Recorder记录、计算训练过程中的所有准确率等指标、acc、metric计算
|  ├─ runningScript.py        # 模型训练基类，使用CE计算图像分类损失
|  ├─ ssl_runner.py           # 自监督预训练的核心训练过程
|  ├─ utils.py                # Early-Stop、映射标签到idx、Onehot、参数计算、FLOPs等等辅助代码
|  └─ wraps.py                # wrapper 编写，使用装饰器来进行 Tensorboard、Logger、运行时间等等的记录
```
