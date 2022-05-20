# ReadMe

@Aiken 2021 实验说明和记录

Try To make structure universal（），编写一个自己的通用的架构，框架化，满足通过不同的model文件和特殊配置文件就能实现不同的模型的一个架构。

只是一个初步的框架集成，还有很多没有完善的地方，目前测试了ResNet18 跑Cifar10，没有什么问题，如果有什么可以改进的地方，或者你实现了一些Feature，**欢迎进行交流**！（私下联系我最好啦！）感谢帮助

> P.S 一些备注
>
> 1. 还有一些可以参数化或者可视化的地方，由于时间关系目前还没有修改，有兴趣的可以自己先添加一下
> 2. 暂时只集成了分类的模块，后续可能会随缘扩展

## 基本规则

### 文件存放规则

代码文件部分：将代码块分成如下的几个部分编写和存放：

- **MAIN-TRAIN-TEST**：
- **DATA-PREPARATION**：
- **MODEL**：
- **LOSS**：
- **UTILS**：
- **LOG VISUALIZATION**：

数据文件部分由如下的几个文档来构成：

- **Log文件夹：**原则：将所有的结果统一的使用board可视化的输出出来，最终的精度得分等等可以编写一个Logging文件来进行特定的输出原则：ConfigFile-RUNTIME-ImportantHYPER的命名形式，具体的超参数，模型复杂度还有相应的运行时间统计，考虑放到tensorboard的输出数据中。
- **Data文件夹**：按照数据集来划分文件夹，其实这个我们应该是在外部存放的，所以我们最好是将这些数据地址放到config文件夹中特定的格式来读取
- **Config文件夹**：
  使用yaml来存放重要的超参数，也就是主要的模型控制，还有模型参数，包括损失函数，优化器，backbone，还有相应的其他部分。

## CODE Detail

数据处理部分分为两个模块：make unbalance & define newclass；

### Loss  Design

对于损失函数的处理方式：将所有自定义的损失函数 `.py`存放在 `./loss`中，将torch等已经被集成的常见loss function，放在其中的 `__init__.py`中，在 `__inti__`中编写 `L_select`函数，进行import管理，通过 `loss_t`动态的返回相应的loss function，具体的操作如下：

```python
from loss import L_select as loss_setter
loss = loss_setter(loss_type)
```

所以我们就i需要对每个写好的loss function在 `__init__`中进行统一的管理

### Model （Layer）Design

对于backbone的处理方式和Loss是大体一致的）

### recorder & Evaluator

#### recorder：

* 按照当前传入的dict 来更新所有评估指标
* 分为两部分输出，一是累加值一是当前值

#### Evaluator：

* 编写call function 对采用的metric计算进行管理 方便return和调用

```
FinalProject
├─ README.md
├─ config
├─ data
│  ├─ __init__.py
│  ├─ collators.py
│  ├─ dataAugment.py
│  ├─ dataUtils.py
│  ├─ datasetGe.py
│  ├─ setTransformer.py
│  └─ unzipImageNet.sh
├─ layers
│  ├─ CausalNormClassifier.py
│  ├─ __init__.py
│  ├─ activations.py
│  ├─ classifier.py
│  ├─ clusters.py
│  ├─ convolution.py
│  ├─ ezConfidence.py
│  └─ warmup.py
├─ loss
│  ├─ NTXent_loss.py
│  ├─ custom_loss.py
│  ├─ evidential_loss.py
│  ├─ longtailed_loss.py
│  ├─ memory_bank.py
│  └─ supContrast_loss.py
├─ main.py
├─ model
│  ├─ EfficientNet.py
│  ├─ ResNet.py
│  ├─ ResNet32.py
│  ├─ ViTs.py
│  └─ small_ResNet.py
├─ run.sh
├─ run_dis.sh
└─ util
   ├─ Visualize.py
   ├─ dis_runner.py
   ├─ lt_runner.py
   ├─ metric.py
   ├─ runningScript.py
   ├─ ssl_runner.py
   ├─ utils.py
   └─ wraps.py

```