# GPU 集群重打包仿真系统 (GPU Cluster Repack Simulation)

## 目录

1. [项目概述](#1-项目概述)
2. [问题建模](#2-问题建模)
3. [系统架构](#3-系统架构)
4. [核心模块详解](#4-核心模块详解)
5. [算法实现](#5-算法实现)
6. [可视化系统](#6-可视化系统)
7. [运行说明](#7-运行说明)
8. [配置参数](#8-配置参数)
9. [输出文件](#9-输出文件)

---

## 1. 项目概述

### 1.1 项目目标

本项目是一个 **GPU 集群碎片整理（Defragmentation）仿真系统**，用于模拟和研究大规模 GPU 集群中资源碎片化问题及其解决方案。核心目标是：

- **建模真实场景**：将 GPU 集群抽象为节点-槽位二维结构，gang 组为资源分配单位
- **碎片化评估**：量化集群碎片化程度（isolated nodes、fully free nodes 等指标）
- **优化求解**：使用模拟退火（Simulated Annealing）算法搜索最优重打包方案
- **动作编排**：将理论最优状态转换为实际可执行的事件驱动调度计划
- **可视化验证**：通过 HTML/SVG 输出直观的集群状态和调度过程

### 1.2 核心技术栈

| 层级 | 技术 |
|------|------|
| 建模 | Python 类层次结构（Cluster/Node/Slot） |
| 优化 | 模拟退火（SA）+ 匈牙利算法 |
| 调度 | 事件驱动编排器（Event-Driven Orchestrator） |
| 可视化 | 纯 HTML/CSS/SVG（无外部依赖） |
| 配置 | Python 原生配置（config.py） |

### 1.3 关键概念

| 概念 | 定义 |
|------|------|
| **Cluster** | 整个 GPU 集群，由 N 个节点组成 |
| **Node** | 单个计算节点，包含 M 个 GPU 槽位 |
| **Slot** | GPU 槽位，gang 的最小分配单元 |
| **Gang** | 占用多个 GPU 槽位的分布式计算任务 |
| **Domain** | 连通域，物理上互联的一组节点 |
| **Fragmentation** | 碎片化，gang 分散在多个节点导致无法容纳大任务 |
| **Reservation** | 槽位预占，优化编排时防止资源冲突 |

---

## 2. 问题建模

### 2.1 集群拓扑建模

```
Cluster (N × M Grid)
├── Node 0: [Slot 0, Slot 1, ..., Slot M-1] ── connections ──► Node 1, Node 3
├── Node 1: [Slot 0, Slot 1, ..., Slot M-1] ── connections ──► Node 0, Node 2
├── ...
└── Node N-1: [Slot 0, Slot 1, ..., Slot M-1]
```

**连通域（Domain）建模**（`state.py:generate_topology_domains`）：

```python
# 1. 创建 L 个基础连通域，每个域至少包含 min_nodes_per_domain 个节点
# 2. 将剩余节点以 50% 概率加入现有域，或成为独立域
# 3. 域内节点按索引顺序连接形成链路
# 4. 更新 cluster.add_connection(u, v) 建立物理连接
```

这种建模确保：
- 每个域内节点互联互通（gang 可以在域内任意分配）
- 不同域之间无直接连接（gang 不能跨域分配）
- 符合真实物理拓扑约束

### 2.2 碎片化状态生成

**初始状态**（`state.py:generate_balanced_state`）：

```python
# 策略：刻意制造碎片化
for node in cluster.nodes:
    # 70% 概率在单节点内分配 1-2 个小 gang
    # 30% 概率不分配（制造孤立空闲槽位）
    
# 如果未达到目标利用率，使用连通分量分配大 gang
# 通过 allocate_gang() 在有足够连续空间的连通分量中分配
```

**碎片化指标**（`cluster.py:get_fragmentation_info`）：

| 指标 | 定义 | 理想值 |
|------|------|--------|
| `isolated_nodes` | 含孤立槽位的节点数 | 0 |
| `fully_free_nodes` | 完全空闲的节点数 | N（全部空闲） |
| `fully_used_nodes` | 完全占用的节点数 | 最大 |
| `total_free` | 总空闲槽位数 | - |
| `total_used` | 总占用槽位数 | N×M×utilization |

### 2.3 目标函数建模

**能量函数**（`optimal_state.py:_compute_system_energy`）：

```
E = -[R_node + R_domain + R_similarity]

其中：
R_node = α × Σ(z_i²)           # 节点空闲率平方和（推动完全空闲节点）
R_domain = μ × Σ(Z_y²)         # 域等效空闲节点平方和（推动域内集中）
R_similarity = w × Σ相似度分数   # 保持 gang 位置稳定性

z_i = node_i.free_slots / total_slots
Z_y = Σ z_i (for node_i in domain_y)
```

**设计原理**：
- 二次项使空闲率趋向 0 或 1（双稳态），避免中间态
- 域级惩罚项鼓励 gang 在同一域内集中
- 相似度项鼓励最小化 gang 迁移距离

---

## 3. 系统架构

### 3.1 模块依赖图

```
┌─────────────────────────────────────────────────────────────┐
│                        main.py                              │
│                   （入口编排层）                             │
└─────────────────┬───────────────────────────────────────────┘
                  │
      ┌───────────┼───────────┬───────────────┐
      ▼           ▼           ▼               ▼
┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────┐
│ cluster  │ │  state   │ │optimal_  │ │visual_state  │
│ .py      │ │  .py     │ │ state.py │ │  .py         │
│          │ │          │ │          │ │              │
│ 底层模型 │ │ 状态生成 │ │ SA优化   │ │ 静态HTML报告 │
└──────────┘ └──────────┘ └────┬─────┘ └──────┬───────┘
                               │              │
                               ▼              │
                     ┌────────────────┐      │
                     │  optimal_plan  │◄─────┘
                     │    .py         │
                     │                │
                     │ 事件驱动编排器  │
                     └───────┬────────┘
                             │
                             ▼
                     ┌────────────────┐
                     │  visual_plan   │
                     │    .py         │
                     │                │
                     │ 交互式HTML可视化│
                     └────────────────┘
```

### 3.2 数据流

```
1. 配置加载 (config.py)
        │
        ▼
2. 集群初始化 (Cluster)
        │
        ▼
3. 拓扑生成 (StateGenerator.generate_topology_domains)
        │
        ▼
4. 碎片化状态生成 (StateGenerator.generate_balanced_state)
        │
        ├────────────────────────────────────────────┐
        ▼                                            ▼
5a. SA 优化 (VolcanoRepackSA)            5b. 直接可视化（不优化）
        │                                            │
        ▼                                            ▼
6. 最优状态 diff 生成                         输出初始状态
        │                                            │
        ▼                                            ▼
7. 事件驱动编排 (EventDrivenOrchestrator)
        │
        ▼
8. HTML 报告生成 (visual_state + visual_plan)
```

---

## 4. 核心模块详解

### 4.1 cluster.py — 集群基础模型

**Slot 类**：
```python
class Slot:
    node_id, slot_id      # 位置标识
    gang_id               # 所属 gang（None = 空闲）
    is_free               # 空闲状态属性
```

**Node 类**：
```python
class Node:
    slots: List[Slot]     # 该节点所有槽位
    connections: Set[int] # 物理连接的节点 ID
    free_slots            # 空闲槽位列表（计算属性）
```

**Cluster 类**：
```python
class Cluster:
    nodes: List[Node]     # 所有节点
    gang_counter          # gang ID 自增器
    gang_grace_time       # gang 优雅退出时间映射
    
    # 核心方法
    add_connection(a, b)             # 添加物理连接
    get_connected_group(node_id)      # BFS 获取连通分量
    get_all_connected_components()    # 获取所有连通分量
    allocate_gang(gang_size)         # 在连通分量内分配 gang
    get_fragmentation_info()          # 获取碎片化指标
```

### 4.2 state.py — 状态生成器

**StateGenerator 类**：

```python
class StateGenerator:
    cluster: Cluster
    domain_nodes: Dict[domain_id, Set[node_id]]  # 域节点映射
    
    generate_topology_domains(num_domains):
        # 1. 将节点随机分配到 num_domains 个基础域（每个至少 2 节点）
        # 2. 剩余节点 50% 加入现有域，50% 成为独立域
        # 3. 域内节点按索引连接
        # 4. 存储 domain_nodes 映射供后续使用
        
    generate_balanced_state(target_utilization):
        # 1. 优先在单节点内制造小碎片（70% 概率分配 1-2 槽位）
        # 2. 用连通分量分配补足目标利用率
        # 3. 返回已分配的 gang_id 列表
```

### 4.3 optimal_state.py — 模拟退火优化器

**RepackStateTracker 类**：维护 SA 过程中的状态追踪

```python
class RepackStateTracker:
    node_z: Dict[node_id, float]    # 节点空闲率 z_i
    domain_Z: Dict[domain_id, float] # 域等效空闲节点数 Z_y
    gang_locations: Dict[gid, Dict]  # gang 位置信息
    
    apply_slot_changes(changes):
        # changes: {node_id: delta_slots}
        # 增量更新 z_i 和 Z_y，返回 ΔR
        
    get_gang_action_cost(gang_id, action_type):
        # 计算动作成本
        # MOVE_LOCAL/GLOBAL: 系数 × (等效迁移利用率)²
        # EVICT: cost_evict × gang_size
```

**VolcanoRepackSA 类**：主优化器

```python
class VolcanoRepackSA:
    solve() -> Dict:
        # 返回：accepted_moves, energy_history, optimal_plan 等
        
    # SA 参数
    alpha=100    # 节点级能量权重
    mu=10        # 域级能量权重
    T_start=500  # 初始温度
    T_end=0.1    # 终止温度
    
    # 动作类型与概率
    prob_local=0.30      # 本地移动
    prob_global=0.30     # 跨域移动
    prob_evict=0.0       # 驱逐
    prob_permute_equiv=0.20  # 节点等价置换
    prob_swap_equiv=0.20     # gang 等价交换
```

**SA 动作详解**：

| 动作 | 描述 | 适用场景 |
|------|------|----------|
| `MOVE_LOCAL` | gang 内部分槽位在同域节点间迁移 | 节点内碎片合并 |
| `MOVE_GLOBAL` | gang 整体迁移到另一域 | 跨域负载均衡 |
| `EVICT` | 驱逐 gang 腾空所有槽位 | 需要完全清空节点 |
| `PERMUTE_EQUIV` | 交换两个等价节点的完整内容 | 等价节点优化 |
| `SWAP_EQUIV_GANG` | 交换两个同大小 gang 的位置 | gang 布局优化 |

**O(1) 回滚机制**：

```python
def _rollback(self, undo_log: List[Tuple]):
    for item in reversed(undo_log):
        if op == 'ASSIGN':
            cluster.nodes[n][s].gang_id = old_gid
        elif op == 'REVERT_Z':
            # 逆向应用空闲率变化
            tracker.apply_slot_changes({n: -delta for n, delta in changes})
        elif op == 'REVERT_LOCATIONS':
            gang_locations[gid]['current_slots'] = old_slots
        # ...
```

### 4.4 optimal_plan.py — 事件驱动编排器

**Cell 状态机**：

```
OCCUPIED ──────────────────────► TERMINATING
   ▲                               │
   │                               │ (grace period)
   │                               ▼
   │                          FREE ◄───── RESERVED
   │                                     ▲
   │                                     │
   └─────────────────────────────────────┘
```

**EventDrivenOrchestrator 核心逻辑**：

```python
class EventDrivenOrchestrator:
    # 布局与状态
    cells: List[List[Cell]]           # 当前集群状态
    initial_layout: Layout            # 初始布局
    target_layout: Layout             # 目标布局
    target_slots: Dict[gid, List[Pos]] # gang 目标槽位
    gang_state: Dict[gid, GangState]   # gang 状态机
    grace_time: Dict[gid, int]         # 优雅退出时间
    
    run():
        while not is_finished():
            # 1. 刷新预占（Reservation）
            self.refresh_reservations()
            
            # 2. 规划动作（优先 start，再 evict）
            actions = self.plan_actions()
            if actions:
                for action in actions:
                    self.submit_action(action)
                continue  # 重新规划
                
            # 3. 处理 release 事件
            events = self.pop_next_events()
            for event in events:
                self.apply_event(event)
            self.refresh_reservations()
```

**动作规划算法**：

```python
def plan_actions(self):
    # 1. 刷新预占槽位
    self.refresh_reservations()
    
    # 2. 构建阻塞图
    blocking_graph = self.build_blocking_graph()
    
    # 3. START 候选（按 score_start 排序）
    #    score = 1000 + 30×size + Σ(10×target_node_fill_gain)
    #    取前 max_parallel_start 个
    
    # 4. EVICT 候选（按 score_evict 排序）
    #    score = base + 180×直接阻塞 + 80×传递阻塞 
    #            + 35×阻塞槽位 + 20×gang大小 - grace_penalty
    #    取前 max_parallel_evict 个
```

**阻塞图与 DAG**：

```python
def build_blocking_graph(self):
    # gang A 阻塞 gang B 当且仅当：
    # A 占用 B 的某个目标槽位（OCCUPIED/TERMINATING/RESERVED）
    
def build_dag_snapshot(self):
    # 构建 E/D → R → S 三层 DAG
    # E/D: evict/delete 节点
    # R: release 节点（gang 退出）
    # S: start 节点（gang 启动）
    # 边: E/D → R (grace), R → S (no duplicate), R → S (free target)
```

**Reservation 机制**：

```python
def refresh_reservations(self):
    # 1. 清除无效预占（非目标 gang、不在目标位置等）
    # 2. 为每个不在目标位置的 gang 预占其所有目标槽位
    # 作用：防止新启动的 gang 抢占正在迁移的 gang 的目标槽位
```

**Stationary Gang 处理**：

```python
def _preserve_stationary_gangs(self, target_layout):
    # 识别初始和目标位置完全一致的 gang
    # 这些 gang 称为 stationary gangs
    # 编排器不会对它们执行 evict 操作
    # 优化：减少不必要的驱逐
```

### 4.5 visual_state.py — 静态报告生成器

```python
class Visualizer:
    def cluster_svg(cluster, title, show_connection):
        # 生成集群拓扑 SVG
        # - 节点行着色表示连通分量
        # - gang 槽位着色（HSL 色彩映射）
        # - 物理连接贝塞尔曲线

def write_repack_report_html(...):
    # 输出包含以下部分的 HTML 报告：
    # - Hero 统计区（运行时、动作数等）
    # - 指标对比卡片（资源使用、碎片节点等）
    # - 集群状态 SVG（前后对比）
    # - 能量收敛曲线
    # - 完全空闲节点曲线
    # - 编排计划时间线
```

### 4.6 visual_plan.py — 交互式编排可视化

```python
class OrchestrationHtmlRenderer:
    # 生成交互式 HTML 报告
    
    # 流程卡片（Flow Cards）
    # - Stage: 提交动作批次
    # - Event: 应用 release 事件批次
    # - State: 状态检查点
    
    # 集群转换视图
    # - 左右对比（before/after）
    # - 贝塞尔曲线箭头标注动作
    # - 鼠标悬停高亮相关 gang
    
    # DAG 视图
    # - E/D → R → S 三层布局
    # - 节点悬停显示传递闭包
    # - 支持缩放拖拽
    
    # JavaScript 交互
    # - Panzoom（滚轮缩放、拖拽平移）
    # - Pod 高亮（hover 显示 gang 所有槽位）
    # - DAG 传递闭包高亮
```

---

## 5. 算法实现

### 5.1 模拟退火算法流程

```
输入: 初始集群状态 S₀, 参数 α, μ, T_start, T_end, max_steps
输出: 最优状态 S*

T ← T_start
cooling_rate ← (T_end / T_start)^(1/max_steps)
S ← S₀, S* ← S₀
E ← energy(S), E* ← E

for step in [1, max_steps]:
    # 随机选择动作类型
    rand ← random()
    if rand < prob_permute_equiv:
        ΔE, ΔC, undo ← apply_equivalence_permutation()
    elif rand < prob_permute_equiv + prob_swap_equiv:
        ΔE, ΔC, undo ← apply_equivalent_gang_swap()
    else:
        action_type ← random()  # {MOVE_LOCAL, MOVE_GLOBAL, EVICT}
        ΔE, ΔC, undo ← apply_action(gang_id, action_type)
    
    E_new ← energy_after_action + ΔC
    
    # Metropolis 准则
    if E_new < E or random() < exp(-(E_new - E) / T):
        accept ← True
        E ← E_new
        if E < E*:
            S* ← current_state
    else:
        rollback(undo)  # O(1) 回滚
    
    T ← T × cooling_rate  # 指数冷却

输出 S* 作为最优状态
```

### 5.2 匈牙利算法（等价映射）

```python
def hungarian_maximize(weights):
    # 将最大化问题转换为最小化：
    # cost[i][j] = max_weight - weights[i][j]
    
    # 标准匈牙利算法 O(n³)
    # 返回最大权匹配
```

**应用场景**：等价节点排序（`optimal_state.py:_apply_equivalence_sort`）

```
等价节点: {N0, N1, N2, N3}（同一连通域内）

目标: 找到节点排列使得相似度最大化

方法:
1. 构建相似度矩阵 W[i][j] = occupancy_score(N_i初始, N_j最优)
2. 调用匈牙利算法求最大权匹配
3. 按匹配结果重排 gang 位置
```

### 5.3 事件驱动调度算法

```
输入: 初始布局 L₀, 目标布局 L*, grace_time 映射
输出: 事件序列 E, 动作序列 A, 快照序列 S

初始化:
    cells ← L₀
    gang_state ← {gang: RUNNING if in L₀ else ABSENT}
    events ← PriorityQueue()  # 按时间排序

主循环:
    while not finished():
        refresh_reservations()
        
        # 规划阶段
        actions ← plan_actions()
        if actions:
            for a in actions:
                submit_action(a)
            capture_snapshot()
            continue
        
        # 等待事件
        batch ← pop_next_events()  # 同时间的多个事件
        for e in batch:
            apply_event(e)
        capture_snapshot()

动作评分:
    score_start(gang) = 1000 + 30×size + Σ(10×node_fill_gain)
    score_evict(gang) = 500 + 180×直接阻塞 + 80×传递阻塞 
                      + 35×阻塞槽位 + 20×gang大小 - grace_penalty
```

---

## 6. 可视化系统

### 6.1 集群 SVG 渲染

**布局计算**（`visual_state.py:render_cluster_svg`）：

```
SVG 尺寸:
    width = left + label_w + slots×cell_w + gaps + right_label_w + connection_w + 24
    height = top + nodes×cell_h + gaps + 38

节点行:
    - 背景矩形着色表示所属连通分量
    - 每个槽位根据 gang_id 分配 HSL 颜色
    - 空闲槽位显示为灰色边框

连接曲线:
    - 贝塞尔曲线连接物理相邻节点
    - 曲线弯曲度与节点间距成正比
```

**gang 颜色算法**：

```python
def gang_color(gang_id):
    hue = (gang_id * 47 + 25) % 360  # 伪随机但确定性的色相
    return f"hsl({hue}, 72%, 82%)"    # 高饱和度、亮色调
```

### 6.2 交互式 DAG

**节点类型**：

| 类型 | 标签 | 层 | 颜色 |
|------|------|-----|------|
| `evict` | E{gang} 或 D{gang} | 0 (E/D) | 白/灰 |
| `release` | R{gang} | 1 (R) | 白/灰 → 蓝（已释放） |
| `start` | S{gang} | 2 (S) | 绿（ready）/黄（pending）/灰（blocked） |

**边类型**：

| 边 | 描述 | 颜色 |
|----|------|------|
| E→R | 驱逐后进入 grace 等待期 | 灰色 |
| R→S | 释放后可以启动 | 灰色 |
| R→S (free target) | 释放后直接可启动 | 深灰/蓝 |

### 6.3 HTML 报告结构

```
<!DOCTYPE html>
<html>
  <head>
    <meta charset="utf-8">
    <style>
      /* 响应式布局 */
      /* 卡片阴影与圆角 */
      /* 指标数字动画 */
    </style>
  </head>
  <body>
    <main class="page">
      <section class="hero">          <!-- 标题统计区 -->
      <section class="card">          <!-- 指标卡片 -->
      <section class="card">          <!-- 集群状态 -->
      <section class="card">          <!-- 收敛曲线 -->
      <section class="card">          <!-- 编排计划 -->
    </main>
    <script>                          <!-- 仅 visual_plan.html -->
      /* Panzoom */
      /* Pod 高亮 */
      /* DAG 传递闭包 */
    </script>
  </body>
</html>
```

---

## 7. 运行说明

### 7.1 快速运行

```bash
cd F:\MyProjects\InnershipProjects\repack_exp
python main.py
```

### 7.2 参数配置

编辑 `config.py`：

```python
# 集群形状
NODES = 10           # 节点数
SLOTS = 10           # 每节点槽位数
DOMAINS = 1          # 连通域数量

# 状态生成
UTILIZATION = None   # None=随机[0.5, 0.8]
SEED = None          # None=使用时间戳

# 优化参数
REPACK = True        # 是否运行 SA 优化
REPACK_STEPS = 30000 # SA 迭代次数
REPACK_ALPHA = 100   # 节点能量权重
REPACK_MU = -1       # -1=自动绑定

# 动作概率
PROB_LOCAL = 0.3
PROB_GLOBAL = 0.3
PROB_EVICT = 0.0
PROB_PERMUTE_EQUIV = 0.20
PROB_SWAP_EQUIV = 0.20

# 编排参数
ENABLE_ORCHESTRATION = True
ORCHESTRATION_MAX_PARALLEL_EVICT = 3
ORCHESTRATION_MAX_PARALLEL_START = 3
```

### 7.3 输出文件

```
optimal_temp/
├── state.json                  # 初始/最优状态矩阵
├── repack_report.html          # SA 重打包报告
├── orchestration_viz.html      # 事件驱动编排可视化
├── orchestration_plan.md       # 编排计划表格
├── dag_merge_insert_demo.html   # 外部 gang 插入演示
└── [缓存的 HTML 片段]
```

---

## 8. 配置参数

### 8.1 SA 参数调优指南

| 参数 | 作用 | 建议范围 | 说明 |
|------|------|----------|------|
| `alpha` | 节点能量权重 | 50-200 | 增大→更追求完全空闲节点 |
| `mu` | 域能量权重 | 5-20 | 增大→更追求域内集中 |
| `REPACK_STEPS` | 迭代次数 | 10000-50000 | 集群越大需越多迭代 |
| `REPACK_ALPHA` | 冷却速度 | 退火进度比 | 固定100 |
| `cost_local` | 本地移动成本 | 200-500 | 增大→减少本地移动 |
| `cost_global` | 跨域移动成本 | 1000-3000 | 增大→减少跨域迁移 |
| `cost_evict` | 驱逐成本 | 5000+ | 增大→避免驱逐 |

### 8.2 编排参数说明

| 参数 | 作用 | 建议范围 |
|------|------|----------|
| `max_parallel_evict` | 每批次最大驱逐数 | 3-8 |
| `max_parallel_start` | 每批次最大启动数 | 3-8 |
| `max_steps` | 编排最大步数 | 1000-5000 |

---

## 9. 输出文件详解

### 9.1 state.json

```json
{
  "initial_state": [
    [0, 0, 1, null, ...],  // Node 0 的槽位 gang 映射
    [2, null, 3, 3, ...],  // Node 1
    ...
  ],
  "optimal_state": [
    // SA 求解后的最优 gang 分布
  ]
}
```

### 9.2 repack_report.html

包含以下面板：
1. **Hero 统计**：运行时、动作数、空闲节点变化
2. **指标对比**：资源使用、碎片节点、完全空闲节点、完全占用节点
3. **集群状态**：Before/After SVG 对比
4. **能量曲线**：SA 能量随迭代收敛图
5. **完全空闲节点曲线**：完全空闲节点数随迭代变化图
6. **编排计划**：动作批次时间线

### 9.3 orchestration_viz.html

交互式可视化，包含：
- **Flow Cards**：每个步骤的集群状态快照
- **Cluster Transition**：before/after 对比 + 动作箭头
- **DAG 视图**：E/D → R → S 三层依赖图
- **Grace Time 分布**：各 gang 优雅退出时间

---

## 附录 A：关键公式

### A.1 节点空闲率

$$z_i = \frac{\text{free\_slots}_i}{\text{total\_slots}_i}$$

### A.2 域等效空闲

$$Z_y = \sum_{i \in \text{domain}_y} z_i$$

### A.3 系统能量

$$E = -\alpha \sum_i z_i^2 - \mu \sum_y Z_y^2 - w \cdot \text{similarity}$$

### A.4 冷却调度

$$T_k = T_0 \cdot \alpha^{\frac{k}{N}}$$

其中 $\alpha = (T_{end} / T_0)^{1/N}$

### A.5 匈牙利算法成本转换

$$\text{cost}_{ij} = \max(\text{weights}) - \text{weight}_{ij}$$

---

## 附录 B：状态转移图

```
┌──────────────────────────────────────────────────────────────┐
│                    EventDrivenOrchestrator                    │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│   ┌─────────┐    submit_start()     ┌─────────┐              │
│   │ ABSENT  │ ───────────────────► │ RUNNING │              │
│   └─────────┘                      └────┬────┘              │
│       ▲                                  │                   │
│       │                                  │ submit_evict()    │
│       │                                  ▼                   │
│       │ apply_event()              ┌───────────┐             │
│       │ (RELEASED)                 │ TERMINATING│            │
│       │                            └─────┬─────┘            │
│       │                                  │                   │
│       │                                  │ grace_time 后     │
│       └──────────────────────────────────┘                   │
│                                                              │
│   GangState: RUNNING → TERMINATING → ABSENT                  │
│                  ↓                   ↓                       │
│              (temp_evict)        (gang_evict)                │
│                  ↓                   ↓                       │
│              RESERVED 槽位被预占      所有槽位变 FREE         │
└──────────────────────────────────────────────────────────────┘
```

---

*本文档于 2026-09-04 | 项目版本: repack_exp*
