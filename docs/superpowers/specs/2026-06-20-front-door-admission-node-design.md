# 设计：对话准入节点（front-door admission node）

> Slice 1。把门口从"只分意图的 IntentRouter"升级成"带记忆、能应付对话表层的准入决策节点"。
> 背景与架构取舍见 [docs/routing-architecture-note.md](../../routing-architecture-note.md)。

## Context

线上连续两个误判暴露了当前"单句分类-路由"的结构性问题：

- 「我完全不会mysql，给我讲懂mysql的概念」→ 被 `QueryPreprocessor` 误判 `out_of_scope`（库内真问题被拒）。
- 「为什么回答不了，你不是有MySQL书吗」（元对话/不满）→ 被路由进 qa、再被 `out_of_scope` 当兜底垃圾桶接走。

经一轮设计讨论得出结论（见 note）：**不是类太少，而是把正交的判断压进一个会长大的 enum，并用 out_of_scope 当兜底**。对话/元问题/反馈这条"会话状态"维度，单句分类根本接不住——它依赖会话历史，不在这句话里。

**这一刀的目标**：在 workflow 门口加一个**带会话记忆的准入决策节点**，先把"对话表层"（寒暄、元问题、对上轮的反馈/不满、指代消解）从主题流程前面摘走；内容问题净化成自包含 query 再下沉现有 qa 流程。**workflow 仍是顶层编排器**（不是 note 早期写的"agent 站外面把 workflow 当工具"的翻转）。

## Goals / Non-goals

**Goals**
- 门口产出四个有界出口之一：`converse` / `clarify` / `dispatch_qa` / `dispatch_study_plan`。
- 元问题/寒暄/对上轮的不满 → `converse`（直接友好回复，不进检索），不再被塞进 out_of_scope。
- 内容问题 → `dispatch_qa(clean_query)`，下游 qa 流程（probe→分类→检索合成）**完全不动**。
- 指代/意图无法从历史定出 → `clarify`（反问），不瞎猜。
- `out_of_scope` 分支回复改成**对话式转场**（更友好），不再机械吐"知识库里暂无相关内容"。

**Non-goals（明确不在这一刀）**
- **不判 scope（库外）**：库外仍由下游 `QueryPreprocessor` 按 probe 召回证据判。门口无检索、不判库存。
- **不修 scope 判据**（"讲懂mysql概念被误判库外"那个 bug 是 judge 判据问题，另开一刀）。
- **不做 preamble、不做会话感知合成**：对抱怨先只做"重构问题→dispatch / 模糊→clarify"，道歉/连续性留给 slice 2 的会话感知合成（单独立项 + faithfulness 把关，避免在顺手改动里削弱结构性防幻觉）。
- **不加结构化 last_outcome 持久化**：repair 靠门口读会话历史（上轮回复文本已在历史里）自然判断，不动 DB/schema。

## 架构

```
DocQueryWorkflow（仍是顶层编排）
  start → route(准入决策节点) ──┬─ dispatch_qa        → PreprocessEvent →（现有 qa 流程，不动）
                               ├─ dispatch_study_plan → StudyPlanEvent
                               ├─ converse            → DirectReplyEvent → Finalize（不检索）
                               └─ clarify             → DirectReplyEvent → Finalize（不检索）
```

门口只判"对话 vs 内容、派给谁"；scope/完整性/结构仍在下游 qa 各轴。一个负责"这句话和对话什么关系"，一个负责"这个 query 关于什么、怎么查"。

## 组件：FrontDoorAgent（决策单元）

新模块 `core/workflow/front_door.py`。**单次 LLM 调用的结构化决策单元**（不是工具循环 FunctionAgent；"agent"是它"带记忆的智能准入门"这个角色）。沿用 `IntentRouter`/`QueryPreprocessor` 既有模式：注入 LLM、`json_object` 模式、Pydantic 校验、失败降级、对外只暴露一个 `run`。

接口：
```python
@dataclass
class FrontDoorDecision:
    action: str          # converse | clarify | dispatch_qa | dispatch_study_plan
    clean_query: str = ""  # dispatch_*：净化后的自包含 query
    reply: str = ""        # converse/clarify：面向用户的话
    reason: str = ""

class FrontDoorAgent:
    def __init__(self, llm: LLM): ...
    async def run(self, original, memory=None, book_titles=None) -> FrontDoorDecision: ...
```

prompt 职责（稳定指令在前、变化输入 history/scope/query 在末尾，命中缓存）：
1. **净化**（并入现 IntentRouter 第一步）：指代消解（读历史 + 选中的书）+ 规范化（纠错/全半角/无歧义缩写），只改形式不改意图。
2. **四选一**（基于会话状态）：
   - `dispatch_qa`：对已入库内容的**具体知识提问**。**红线**：哪怕你自己知道答案，也绝不在这里作答——一律下沉检索。
   - `dispatch_study_plan`：基于某书生成学习计划/路线。
   - `converse`：寒暄/致谢/问你是谁能做什么这类元问题，**以及对上一轮的反馈/质疑/不满**（读历史能看到上轮回复，包括拒答那句）。`reply` 给自然回复。
   - `clarify`：明显指会话里某物但**历史里定不出所指**（落很早/有歧义）。`reply` 给一句反问，能列候选就列。
3. **对抱怨的判据**：意图清楚（"我要入门""你答错了，应是 Y"）→ `dispatch_qa(重构出的 clean_query)`；意图模糊（光一句"你逗我呢"）→ `clarify`，别猜着重构。

复用 `format_history` / `format_scope` / `_strip_fences`（见下 IntentRouter 处置）。

降级（绝不阻塞，内容默认下沉）：
- 解析失败 / 空 content / 非法 action（Pydantic 拒）→ `dispatch_qa(原 query)`。
- `dispatch_*` 但 clean_query 空 → 回退原 query。
- `converse`/`clarify` 但 reply 空 → 通用兜底文案（非空）。

## Workflow 接线（DocQueryWorkflow）

- `route` step 改调 `FrontDoorAgent`（替掉 `IntentRouter`），按 action 返回事件：
  - `dispatch_qa` → `PreprocessEvent`（store `clean_query`，下游不动）
  - `dispatch_study_plan` → `StudyPlanEvent`
  - `converse` / `clarify` → 新 `DirectReplyEvent(reply, action)`
- 新增 `DirectReplyEvent` + `direct_reply_branch`：直接把 `reply` 当答案 → `FinalizeEvent`，不进 probe/检索。**替掉写死套话的 `ChitchatEvent`/`chitchat_branch`**。
- `out_of_scope_branch`：回复改对话式转场（友好承认 + 邀请换个问法/指向库内能帮的方向），替掉"知识库里暂无与该问题相关的内容。"。
- `finalize`：照旧把答案（含 converse/clarify 的 reply）写回会话 memory；metadata 附 `action`（替代/并入原 `intent`）供观测。

## IntentRouter 处置

被 `FrontDoorAgent` 取代：
- 把复用 helper（`format_history` / `format_scope` / `_strip_fences` / `MAX_HISTORY_MSGS`）迁入 `front_door.py`（或共享处）。
- 迁移 `route` step 后，删除 `IntentRouter` 类与 `tests/test_intent_router.py`、删除 `ChitchatEvent`/`chitchat_branch`。
- 实现前先 grep 确认无其它 importer（eval 路径走 `DocQueryWorkflow` 整体、不直接依赖 `IntentRouter`）。

## 测试

- **FrontDoorAgent 单测**（mock LLM，镜像 test_intent_router 模式）：四出口解析；净化输入透传（history/选中书进 prompt）；降级（解析失败/空/非法 action → dispatch_qa+原 query；空 clean_query 回退；空 reply 兜底）。
- **workflow 层**：route 按 action 分发到正确事件；converse/clarify → DirectReply 的 reply 成为答案；out_of_scope 新文案。
- **评测不受影响**：golden 全是 qa → `dispatch_qa` → preprocess，分类准确率/指标口径不变。冒烟一次确认链路通。

## 命名

类名 `FrontDoorAgent`（贴讨论里"起点/准入 agent"的叫法）。若倾向与现有决策单元命名一致（`*Router`/`*Preprocessor`），可改 `FrontDoorRouter`——评审时定。

## 后续刀（不在本设计）

- **会话感知合成（slice 2）**：合成器吃[压缩历史(框架/语气/连续性/道歉)]+[chunk(唯一事实源)]，prompt 立"事实只来自 chunk"硬规矩，用 faithfulness 当门槛指标把关。道歉/连续性在这刀自然拿到。
- **scope 判据修正**：out_of_scope 以"主体技术实体"为准、深度/角度不匹配不算库外。
