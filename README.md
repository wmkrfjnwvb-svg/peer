# peer：数据存在模型权重里的数据库

数据库的数据转成张量（**数据权重**），由一组**精确算子专家**执行查询，大模型只负责**选择**：调用哪些算子、指向哪张表哪一列。数据不经过模型生成，所以只要选择正确，结果就和 SQLite 逐条一致；不支持的写法直接报错，不会返回猜测的结果。

```
SQL ───────────────────────┐
中文/英文问题 → [模型] → SQL ─┴→ [模型] → 执行计划 → [精确算子 × 数据权重] → 结果
                                       (算子 + 指针)       (确定性张量运算)
```

## 目录

| 路径 | 内容 |
|---|---|
| `peerdb/vec.py` | 列的张量表示，SQLite 的类型、比较、亲和性规则 |
| `peerdb/store.py` | 数据权重：SQLite → 张量，保存 / 加载为单个文件 |
| `peerdb/plan.py` | 执行计划（模型输出）的格式、解析、打印 |
| `peerdb/executor.py` | 算子专家：scan / join / leftjoin / filter / group / sort / limit / project / distinct / 集合运算 |
| `peerdb/compiler.py` | 教师编译器：SQL → 执行计划，**只用于生成训练标签** |
| `scripts/prepare_data.py` | 生成训练数据：编译 → 执行 → 和 SQLite 对拍，只保留结果完全一致的样本 |
| `train.py` | 微调 Qwen：SQL→计划、问题→SQL 两个任务 |
| `eval.py` | 端到端评测，带置信度阈值（召回率控制） |
| `demo.py` | 交互演示 |
| `tests/` | 执行器和 SQLite 的对拍测试 |

## 1. 安装

```bash
pip install -r requirements.txt
```

## 2. 下载数据

| 数据集 | 规模 | 协议 | 下载 |
|---|---|---|---|
| Spider | 1.1 万条，200 个 SQLite 库 | CC BY-SA 4.0 | `HAL-9001/spider-databases`（HF，带数据库） |
| SynSQL-2.5M | 254 万条，1.6 万个 SQLite 库 | Apache 2.0 | `seeklhy/SynSQL-2.5M`（HF） |
| BIRD | 1.2 万条，95 个库（33 GB） | CC BY-SA 4.0 | bird-bench.github.io |

```bash
mkdir -p data/raw && cd data/raw
# Spider（206 MB）
wget https://huggingface.co/datasets/HAL-9001/spider-databases/resolve/main/spider_data.zip
unzip spider_data.zip -d spider && rm -rf spider/__MACOSX
# SynSQL-2.5M（数据库 54 MB，问答 9.4 GB）
mkdir synsql && cd synsql
wget https://huggingface.co/datasets/seeklhy/SynSQL-2.5M/resolve/main/databases.zip && unzip databases.zip
wget https://huggingface.co/datasets/seeklhy/SynSQL-2.5M/resolve/main/data.json
cd ../../..
```

## 3. 生成训练数据（编译 + 和 SQLite 对拍）

```bash
python scripts/prepare_data.py spider --root data/raw/spider/spider_data --out data/spider
python scripts/prepare_data.py synsql --root data/raw/synsql --out data/synsql --limit 500000
```

每条记录的 `status`：

| status | 含义 | 用于 SQL→计划训练 |
|---|---|---|
| `ok` | 执行计划的结果和 SQLite 完全一致 | ✅ |
| `ok_tie_order` | 行相同，只是排序键并列时的先后不同（SQL 标准未规定） | ✅ |
| `unsupported` | 用到了还没实现的特性，明确报错 | ❌ |
| `ambiguous_bare_column` | 聚合查询里选了既没分组也没聚合的列，SQL 标准未规定取哪一行 | ❌ |
| `mismatch` | 结果不一致（执行器 bug，应为 0） | ❌ |

问题→SQL 任务使用所有能在 SQLite 上执行的样本。

## 4. 训练（4 × RTX 4090 24GB）

```bash
# Qwen2.5-Coder-1.5B 全参数微调，FSDP 把参数和优化器状态分到 4 张卡
torchrun --nproc_per_node 4 train.py \
  --model Qwen/Qwen2.5-Coder-1.5B \
  --data data/spider --data data/synsql \
  --out runs/peer-1.5b --fsdp \
  --batch 4 --grad-accum 4 --epochs 3 --lr 2e-5
```

- 有效 batch = 4 卡 × 4 × 4 = 64。
- 显存不够就把 `--batch` 降到 2、`--grad-accum` 升到 8；或者换 `--model Qwen/Qwen2.5-Coder-0.5B`（0.5B 不加 `--fsdp` 也能放下）。
- 只练 SQL 路径：`--tasks sql2plan`。

## 5. 评测

```bash
# SQL 路径：给定 SQL，模型生成执行计划
python eval.py --model runs/peer-1.5b/final --data data/spider --db-root data/raw/spider/spider_data --mode sql
# 自然语言路径：问题 → SQL → 执行计划
python eval.py --model runs/peer-1.5b/final --data data/spider --db-root data/raw/spider/spider_data --mode nl
```

输出执行准确率，以及不同置信度阈值下的"回答比例 / 准确率 / 错误条数"。阈值以下拒答，这就是召回率控制。

## 6. 演示

```bash
python demo.py --model runs/peer-1.5b/final \
  --db data/raw/spider/spider_data/database/concert_singer/concert_singer.sqlite
```

第一次运行把 SQLite 导入成 `*.weights.pt`，之后所有回答都来自这个数据权重文件。

## 已支持 / 未支持

已支持：SELECT、WHERE、INNER / LEFT / CROSS JOIN、GROUP BY、HAVING、ORDER BY、LIMIT / OFFSET、DISTINCT、UNION / UNION ALL / INTERSECT / EXCEPT、IN / NOT IN（列表和子查询）、标量子查询、FROM 子查询、CTE（非递归）、LIKE、BETWEEN、IS NULL、CASE、`||`、四则运算和取模、COUNT / SUM / AVG / MIN / MAX（含 DISTINCT）、lower / upper / length / abs / coalesce。SQLite 的类型亲和性、NULL 三值逻辑、BINARY 排序规则、聚合查询的裸列规则都按 SQLite 实现。

未支持（会明确报错）：窗口函数、日期函数、CAST、关联子查询、EXISTS、递归 CTE、RIGHT / FULL JOIN、GROUP_CONCAT、写操作（INSERT / UPDATE / DELETE）。

## 测试

```bash
python -m pytest -q tests
```
