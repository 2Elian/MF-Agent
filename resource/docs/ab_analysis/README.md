# 检索与意图识别模型实验报告

本文档总结了基于金融多源异构数据的检索问题系统的多阶段实验结果与分析。数据来自四类精心构建并带标签的数据集，可用于 **检索精度评估** 与 **意图识别模型的辅助评估**。

---

## 一、纯 Embedding 检索实验

### 1. 实验设定

该实验仅基于 **Embedding 检索器**，不包含倒排索引（ES）和图谱检索器。  
假设意图类别完全正确（即类别误判不会影响检索精度）。为了简单起见，Embedding的消融实验只基于细节类别来判定。

- 数据规模：细节类测试数据共 62 条  
- 平均每个查询关联的 GT_ID 数：1.89 ，min=1, max=12
- 评估指标：Recall@3、Precision@3、F1、Top-3（业务指标）

> **Top-3 指标说明**  
> 当检索结果排名前三的知识块中存在任意一个正确答案时，视为该查询检索正确。  
> 因此，MRR（Mean Reciprocal Rank）指标在此业务中意义不大。本文评估暂不考虑MRR指标。

> **低精确率原因**  
> 每个查询对应的 GT_ID 数较少（通常一个知识块即可回答），导致 P@3 较低。

---

### 2. 实验结果

| 模型配置 | R@3 | P@3 | F1 | Top-3 |
|-----------|------|------|------|--------|
| Baseline(FinBERT2-small) | 62.19 | 35.16 | 44.92 | 67.92 |
| 微调（FT） | 74.92 | 40.19 | 52.32 | 79.25 |
| 蒸馏 + neg_num=1 + Hard Neg 随机采样 | 79.41 | 46.99 | 59.04 | 86.38 |
| 蒸馏 - On-Policy + neg_num=10 + Hard Neg 采样 | **81.06** | **50.93** | **62.56** | **87.47** |

注意：蒸馏采用的是large作为老师模型，small作为学生模型，蒸馏方法采用的是正向相似度分数蒸馏。

---

### 3. Hard Negative 构造方法

在金融业务线中，Hard Neg 数据的构造流程如下：

1. 构造数据格式：
   ```json
   {
     "query": "",
     "cls": "detail",
     "title": "",
     "ans_id": [
       "74eef74e7a3058ca3a8ca4bb0548c12f",
       "b8b689ef9cecc19bc49e048b7049d6c8",
       "b1798348459131c3d8b682632572e12d",
       "d0692b7d72712d1a0232ec7bed6e00e4"
     ],
     "entity": {},
     "keywords": [] # 此阶段的关键字==实体名称 因为意图识别模型不参与本实验中 因此实际的keyword无法获取
   }
   ```

2. 基于 Base 模型进行检索，取 Top-20 结果：
   - 统计其中的负样本数量（20 - k，k 为正样本数量）。
   - 若负样本的 `title` 与 Query 的 `title` 对应，则视为 **Hard Neg**。
   - 若 `keywords` 重叠数 ≥ 3，同样视为 **Hard Neg**。

3. 最终得到的 Embedding 训练数据：
   ```json
   {
     "query": "string",
     "pos": ["id1", "id2"],
     "neg": ["id3", "id4"]
   }
   ```
   样本数量约为 200（细节类的训练数据为200条），经 **Query 改写（基于 LLM）** 与 **Review 过滤** 后扩充至 800+。

---

### 4. 结论与分析

1. **Hard Neg 采样必要性**  
   可提升约 1–2 个百分点的精度。

2. **微调 vs 蒸馏**  
   在金融场景下，蒸馏模型显著优于普通微调模型，因为金融bert模型对金融这些词的感知度更强。

3. **负样本数量影响**  
   - 当 `neg=1` 时，Loss 曲线震荡较大，可收敛至 >5e-4 水平。  
   - 当 `neg=10` 时，Loss 曲线更平稳，可收敛至 <5e-4。

4. **LoRA vs 全参**  
   差异不明显。

5. **Off-Policy 蒸馏**  
   参考 Qwen3-Embedding 技术报告，其后训练阶段采用 **Off-Policy 蒸馏方法**，理论可行，后续任务中计划尝试。

---

## 二、多策略检索器精度

实验假设意图类别完全正确，分别测试 ES 与 Embedding 的多策略组合。  
（所有检索器构建时均不带 `title` 信息。）

### 细节类检索器结果

| 检索策略 | R@3 | P@3 | F1 | Top-3 |
|-----------|------|------|------|--------|
| BestEmbedding | 81.06 | 50.93 | 62.56 | 87.47 |
| Embedding + ES | 81.69 | 51.45 | 63.14 | 89.01 |
| ES + Embedding | **82.04** | **52.01** | **63.66** | **89.65** |

---

## 三、意图识别模型实验
由于有很多新的token要加，相较于原模型词表会改变，所以我们肯定是要先训一个SFT model，来适配E和L层(Embedding层和Lm Head层)，于是开展了如下实验：

### 3.1 冷启动sft阶段实验

在冷启动模型训练中，我们要思考一个问题：用什么方法来训这个模型？

思考1：我们这个阶段只是想让模型适应一下新任务，换句话来说就是让模型适应一下新token的分布。如若此时直接用全量模型训的话，可能会导致中间层的分布坍塌（这里的意思是说，本来我只是想调整一下embedding层和lm head层来适配新任务，可是如果全参训练的话，那中间层也会随之改变很多），这一定会导致模型的通用能力下降。而我个人觉得：一个好的冷启动模型不仅仅是适配新token，而且还要保持原有的能力，这样在后续RL训练的时候才会更稳定一些。

思考2：既然不能使用全参更新模型，相比于先冻结中间层训练E和L层再训练全参模型，lora来的更直接一些。所以这个阶段我们就保持lora训练即可。

思考3：lora训练 除了E和L层，中间层哪些是可以不动的？暂未完成实验，后续有时间补上

思考4：lora训练是否要混入通用数据来保证通用能力呢？


**先来看如下的评估结果**：
<details>
<summary><b>训练数据示例</summary>

一条训练数据原子类型如下(最终映射成shareGPT形式进行训练，sft的ans是基于我们自己打的线索标签gt，使用上下文学习采样得到的答案(deepseek-满血api)。)

    "query": "介绍一下上海浦东发展银行公司网上银行业务手册（三）中的网银交易授权",
    "entity": ["上海浦东发展银行", "网上银行业务手册", "网银交易授权"], # 来自ner模型识别
    "gt": {"title":"上海浦东发展银行公司网上银行业务手册（三）", "keywords": ["网银交易授权"], "cls": "detail"}, # 来自标注
    "sft_ans": "<think>这个问题明确要求介绍“网银交易授权”，属于对单一细节的查询，没有涉及多个意图或多跳推理。因此，这条query属于<question_type>detail</question_type>。此外我注意到query中有个（三），这应该是一个标题类型才会有的，所以结合query的隐藏信息可以得到<title_name>上海浦东发展银行公司网上银行业务手册（三）</title_name>。接下来，让我再来确定一下实体信息为：<ner_entities>["上海浦东发展银行", "网上银行业务手册", "网银交易授权"]</ner_entities>，从实体信息和query信息来看，可以看出此问题重点在于询问网银交易授权的问题，因此，可以可以得到关键词信息<keywords>["网银交易授权"]</keywords>。</think><question_type>detail</question_type><title_name>上海浦东发展银行公司网上银行业务手册（三）</title_name><ner_entities>["上海浦东发展银行", "网上银行业务手册", "网银交易授权"]</ner_entities><keywords>["网银交易授权"]</keywords>"

shareGPT格式：

    {
    "conversations": [
        {
            "role": "system",
            "content": "# 你是一个专业的查询解析助手。你的核心任务是对用户输入的查询进行深度分析，并将其结构化为标准的机器可读格式。# 请严格遵循以下思考和分析步骤：1.判断问题类型： 确定用户的查询意图是属于 detail（询问细节）、compare（进行比较）、procedural（询问流程）还是其他类型。2.推断来源标题： 分析查询上下文，识别并提取出用户问题所指向的文档或资料来源的完整标题。3.识别命名实体： 找出查询中所有关键的命名实体，如组织机构、人名、地点、专业术语等。4. 提炼核心关键词： 从查询中提炼出最核心、最能代表用户搜索意图的一个或多个关键词。5.如果是多意图任务需要输出拆解后的意图数量以及每个意图的子问题和子关键词列表。如果是多跳和推理类型，同样需要进行问题的拆解。# 输出格式：以细节类型为例，你必须将最终分析结果包裹在 <think> 标签内，并严格使用以下指定的XML标签进行输出：<think>[你的逐步推理过程]</think><question_type>[问题类型]</question_type><title_name>[来源标题]</title_name><ner_entities>[实体列表]</ner_entities><keywords>[关键词列表]</keywords>"
        },
        {
            "role": "user",
            "content": "介绍一下上海浦东发展银行公司网上银行业务手册（三）中的网银交易授权"
        },
        {
            "role": "assistant",
            "content": "<think>这个问题明确要求介绍“网银交易授权”，属于对单一细节的查询，没有涉及多个意图或多跳推理。因此，这条query属于<question_type>detail</question_type>。此外我注意到query中有个（三），这应该是一个标题类型才会有的，所以结合query的隐藏信息可以得到<title_name>上海浦东发展银行公司网上银行业务手册（三）</title_name>。接下来，让我再来确定一下实体信息为：<ner_entities>["上海浦东发展银行", "网上银行业务手册", "网银交易授权"]</ner_entities>，从实体信息和query信息来看，可以看出此问题重点在于询问网银交易授权的问题，因此，可以可以得到关键词信息<keywords>["网银交易授权"]</keywords>。</think><question_type>detail</question_type><title_name>上海浦东发展银行公司网上银行业务手册（三）</title_name><ner_entities>["上海浦东发展银行", "网上银行业务手册", "网银交易授权"]</ner_entities><keywords>["网银交易授权"]</keywords>"
        }
    ]
    }


新的token：

    <question_type> </question_type>
    <title_name> </title_name>
    <ner_entities> </ner_entities>
    <keywords> </keywords>
    <num> </num>
    <information> </information>

    注意：实际训练的时候，我们还做了一道转换，因为我们打标签的时候都把特殊token打成了<></>的形式。而实际上qwen的special token是<|xx_start|><|xx_end|>的形式。

思路流程如下：

    <question_type> cls str </question_type>
    <document_name>str</document_name>
    <ner_entities>List</ner_entities>
    if cls == detail:
        <keywords> </keywords>
    elif cls == multi_intent:
        <num> </num>
        <information>
        List[Dict]{
                "question": "拆解后的问题1",
                "entity": ["实体A", "实体B"],
                "keywords": ["关键词+实体内容"]}
        </information>
    elif cls == multi_hop:
        <step_num> </step_num>
        <information>
        List[Dict]{
                "question": "拆解后的问题1",
                "entity": ["实体A", "实体B"],
                "keywords": ["关键词+实体内容"]}
        </information>
    elif cls == reasoning:
        <step_num> </step_num>
        <information>
        List[Dict]{
                "question": "拆解后的问题1",
                "entity": ["实体A", "实体B"],
                "keywords": ["关键词+实体内容"]}
        </information>

</details>
评估维度：

1. 通用能力与数学能力是否下降  
2. 分类准确性与输出 `title` 的模糊匹配准确率

| 数据集评估 | Only Domain + LoRA | Only Domain + 全量 | 混合源数据 + LoRA |
|-------------|--------------------|--------------------|----------------------|
| MMLU-PRO | 65.7(-3.9) | 54.3(-15.3) | 69.1(-0.5) |
| GPQA | 60.0(-2.0) | 51.7(-10.3) | 61.8(-0.2) |
| Domain-cls | 0.87 | 0.89 | 0.90 |


**结论分析**
1. 全量 vs lora
   
   全量会对中间层的分布带来较大的变化，虽然领域适配能力还可以，分类能力能达到0.89，但对通用能力打击太大。


2. 是否需要混入通用数据保证通用能力？

    需要，按比例混入了域数据量的5倍通用数据+2倍数学数据。效果达到最好。

![Lora Loss曲线](resource/images/lora_loss.png)

---

### 3.2 RLVR 阶段实验

在冷启动模型基础上进行 RLVR 训练，指标如下：

| 评估项 | Have Cold Start Model |
|----------|----------------|
| Best loss (600 steps) | 0.37 |
| Domain-cls | 0.99 | 
| Domain-title | 优 | 
| Domain-information | 优 | 

---

### 3.3 Pipeline 精度评估

以 **ES + Embedding 策略** 为 baseline，对比意图识别模型效果。  
意图识别模型除输出意图类别外，还会输出指导性 `query_title` 与 `keywords`，用于三阶段检索与排序。

| 检索策略 | R@3 | P@3 | F1 | Top-3 |
|-----------|------|------|------|--------|
| Baseline | 82.04 | 52.01 | 63.66 | 89.65 |
| Our Intention Recognition Model | 93.85 | 60.83 | 73.82 | 97.12 |
| Our Intention Recognition Model + keywords filter | **94.91** | **61.17** | **74.39** | **98.35** |
