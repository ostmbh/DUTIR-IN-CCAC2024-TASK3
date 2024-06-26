# DUTIR-IN-CCAC2024-TASK3
# 《基于用户画像的中文情绪分类》评测报告
## 任务描述
 给定一个用户画像（包括用户所在的地区、性别、关注人列表、发表的历史文本等）以及一个用户文本，参赛模型需判断用户文本所表达的情绪，输出的情绪是下面五种情绪之一：喜、哀、惊、恐、怒。
### 数据样例：

##### 输入：
**用户画像**
    - 地区：上海
    - 性别：女
    - 关注的人列表：...
    - 用户发表的历史文本列表：...
**推文**
    - 刚刚买的新裙子，真的超级好看，我简直爱死了！
##### 输出：
喜

评价指标： Macro-F1 score

$F1_{macro} = \frac{1}{N} \sum_{i=1}^{N} F1_i$

其中，$N$是类别数量，$F1_i$是第i类的F1分数

## 任务定义:
输入:
1. 待预测文本$x$
2. 用户数据$u$
    - 用户ID $u_{id}$
    - 用户信息 $u_{info}$：位置，性别
    - 用户历史推文 $u_{his} = \{ p_1, p_2, ... p_N \}$
    - 用户昵称 $u_{name}$
    - 用户社交网络：关注类标，被关注列表...

输出:
    用户对于文本 $x$ 的情感 $y = f(x, u)$
## baselines
### 1.基于bert-base-chinese预训练模型进行微调
参考[Chinese-Text-Classification](https://github.com/shuxinyin/Chinese-Text-Classification/tree/master/bert_classification)，对bert-base-chinese进行微调。
```
baseline/bert-base-chinese/raw data/100 epoch/lr 0.005
loss: 0.7072446474432945
              precision    recall  f1-score   support

           喜     0.8195    0.8926    0.8545       773
           哀     0.7444    0.8200    0.7804       650
           惊     0.6500    0.2989    0.4094        87
           恐     1.0000    0.0000    0.0000        42
           怒     0.6857    0.4768    0.5625       151

    accuracy                         0.7757      1703
   macro avg     0.7799    0.4977    0.5214      1703
weighted avg     0.7748    0.7757    0.7565      1703
```
### 2.使用zero-shot对qwen7b模型直接进行问答
```
llm zero-shot/qwen1.5-7b/5 epoch
              precision    recall  f1-score   support

           喜     0.6372    0.7215    0.6768       650
           哀     0.8048    0.6934    0.7450       773
           惊     0.5370    0.9139    0.6765       151
           恐     0.4000    0.3333    0.3636        42
           怒     0.7778    0.0805    0.1458        87

    accuracy                         0.6835      1703
   macro avg     0.6314    0.5485    0.5215      1703
weighted avg     0.7057    0.6835    0.6728      1703
```
## method
基于[Firefly](https://github.com/yangjianxin1/Firefly)对[Qwen2](https://github.com/QwenLM/Qwen2)模型进行微调，采用如下的prompt策略，并使用硬投票进行汇总：
### 1.SFT
训练参数及数据见/task3/models/sft
```
llm sft/qwen1.5-7b
              precision    recall  f1-score   support

           喜     0.9440    0.8815    0.9117       650
           哀     0.9535    0.9806    0.9668       773
           惊     0.7939    0.8675    0.8291       151
           恐     0.7500    0.6429    0.6923        42
           怒     0.7500    0.8621    0.8021        87

    accuracy                         0.9184      1703
   macro avg     0.8383    0.8469    0.8404      1703
weighted avg     0.9203    0.9184    0.9184      1703
```
### 2.嵌入性别SFT
训练参数及数据见/task3/models/sft_with_gender
```
llm sft withg/qwen1.5
              precision    recall  f1-score   support

           喜     0.9282    0.9154    0.9218       650
           哀     0.9641    0.9715    0.9678       773
           惊     0.8037    0.8675    0.8344       151
           恐     0.7812    0.5952    0.6757        42
           怒     0.8068    0.8161    0.8114        87

    accuracy                         0.9237      1703
   macro avg     0.8568    0.8332    0.8422      1703
weighted avg     0.9236    0.9237    0.9232      1703
```
### 3.嵌入地区SFT
训练参数及数据见/task3/models/sft_with_location
```
llm sft withl/qwen1.5
              precision    recall  f1-score   support

           喜     0.9332    0.8385    0.8833       650
           哀     0.9688    0.9625    0.9656       773
           惊     0.6509    0.9139    0.7603       151
           恐     0.8148    0.5238    0.6377        42
           怒     0.6786    0.8736    0.7638        87

    accuracy                         0.8955      1703
   macro avg     0.8093    0.8224    0.8021      1703
weighted avg     0.9084    0.8955    0.8976      1703
```
### 4.嵌入性别及地区SFT
训练参数及数据见/task3/models/sft_with_location_and_gender
```
llm sft withlg/qwen1.5
              precision    recall  f1-score   support

           喜     0.8920    0.9400    0.9154       650
           哀     0.9630    0.9767    0.9698       773
           惊     0.9569    0.7351    0.8315       151
           恐     0.6970    0.5476    0.6133        42
           怒     0.8353    0.8161    0.8256        87

    accuracy                         0.9225      1703
   macro avg     0.8688    0.8031    0.8311      1703
weighted avg     0.9223    0.9225    0.9206      1703
```
### 5.嵌入标签解释SFT
训练参数及数据见/task3/models/sft_with_def

使用如下prompt格式对大模型指令微调：

请从”喜“、”哀“、”惊“、”恐“、”怒“五种情感中对以下推文进行情感分类，其中情感定义如下：

”喜“：快乐；高兴；愉悦；

”哀“：悲痛；悲伤；

”惊“：吃惊；惊奇；惊喜；惊怒；感慨

”恐“：严重害怕；恐惧；担忧

”怒“：发怒；生气；恼怒；愤怒；

```
llm sft with def/qwen1.5
             precision    recall  f1-score   support

           喜     0.9112    0.9477    0.9291       650
           哀     0.9853    0.9534    0.9691       773
           惊     0.9389    0.8146    0.8723       151
           恐     0.7027    0.6190    0.6582        42
           怒     0.6937    0.8851    0.7778        87

    accuracy                         0.9272      1703
   macro avg     0.8464    0.8440    0.8413      1703
weighted avg     0.9311    0.9272    0.9278      1703
```
### 6.嵌入标签解释+性别
训练参数及数据见/task3/models/sft_with_defg
```
              precision    recall  f1-score   support

           喜     0.9552    0.8200    0.8825       650
           哀     0.9738    0.9625    0.9681       773
           惊     0.7035    0.9272    0.8000       151
           恐     0.6889    0.7381    0.7126        42
           怒     0.5839    0.9195    0.7143        87

    accuracy                         0.8972      1703
   macro avg     0.7811    0.8735    0.8155      1703
weighted avg     0.9158    0.8972    0.9012      1703
```
### 7.嵌入标签解释+地区
训练参数及数据见/task3/models/sft_with_defl
```
              precision    recall  f1-score   support

           喜     0.9248    0.9077    0.9161       650
           哀     0.9714    0.9677    0.9695       773
           惊     0.8947    0.7881    0.8380       151
           恐     0.7143    0.7143    0.7143        42
           怒     0.6667    0.9195    0.7729        87

    accuracy                         0.9201      1703
   macro avg     0.8344    0.8595    0.8422      1703
weighted avg     0.9249    0.9201    0.9212      1703
```
### 8.嵌入标签解释+性别+地区
训练参数及数据见/task3/models/sft_with_deflg
```
              precision    recall  f1-score   support

           喜     0.9053    0.9123    0.9088       650
           哀     0.9774    0.9521    0.9646       773
           惊     0.8075    0.8609    0.8333       151
           恐     0.8846    0.5476    0.6765        42
           怒     0.7222    0.8966    0.8000        87

    accuracy                         0.9160      1703
   macro avg     0.8594    0.8339    0.8366      1703
weighted avg     0.9195    0.9160    0.9162      1703
```
## How to use
克隆本repo至你的环境
```
git clone https://github.com/ostmbh/DUTIR-IN-CCAC2024-TASK3.git
cd DUTIR-IN-CCAC2024-TASK3
pip install -r requirements.txt
```
克隆Firefly至你的环境
```
git clone https://github.com/yangjianxin1/Firefly.git
cd Firefly
pip install -r requirements.txt
cd ..
```
下载模型(请在download.py中更改模型保存路径)
```
export HF_ENDPOINT=https://hf-mirror.com
python download.py
```
更改对应train_args中模型及文件路径，并训练
```
cd Firefly
python train.py --train_args_file ../models/your_args_path/train_args.json
```
评估结果（请修改adapter中模型等文件路径）
```
cd ..
python eval.py
python vote.py
```
## Acknowledgements

- 本项目使用了[Chinese-Text-Classification](https://github.com/shuxinyin/Chinese-Text-Classification/tree/master/bert_classification)，[Firefly](https://github.com/yangjianxin1/Firefly)，以及[Qwen2](https://github.com/QwenLM/Qwen2),感谢项目作者的付出！
