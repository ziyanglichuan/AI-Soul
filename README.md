# AI-Soul: 虚拟角色交互系统 (AI-Soul: Virtual Character Interaction System)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![Ollama](https://img.shields.io/badge/Ollama-blueviolet?logo=ollama)
![ChromaDB](https://img.shields.io/badge/ChromaDB-purple?logo=chroma)
![Diffusers](https://img.shields.io/badge/Diffusers-orange?logo=huggingface)
![Streamlit](https://img.shields.io/badge/Streamlit-red?logo=streamlit)

`AI-Soul` 是一个探索“AI-Native”游戏体验的技术原型。


我们致力于创造“角色灵魂”——他们会记忆、反思、回应，并与玩家和同伴共同成长。这个仓库包含了实现 `AI_Soul` 类的核心代码、AI 间的交互逻辑以及用于演示的 Streamlit 原型。


## ✨ 核心特性 (Core Features)

* **动态个性 AI (Pillar 1):**
    * **双重记忆系统:** AI 拥有“事件记忆”（发生了什么）和“反思记忆”（AI的内心独白），均存储于 `ChromaDB`。
    * **思维链 (CoT) 反思闭环:** AI 的“内心独白”会自动存入“反思记忆”，形成“思考 -> 沉淀 -> 影响未来思考”的进化闭环。
    * **动态情感与好感度:** AI 会基于其“人设”，通过 LLM “元评估”来动态决定对玩家或其他 AI 的好感度变化。

* **AI 驱动的视觉 (Pillar 2):**
    * **AIGC 角色创建:** 基于 AI 的 `base_persona` (核心人设)，使用 `diffusers` (Stable Diffusion) 实时生成“手办”风格的立绘。

* **智能场景交互 (Pillar 3):**
    * **AI 间自主对话 (C2C):** AI 之间会基于彼此的记忆和好感度，主动发起或回应 conversation。
    * **旁观者吐槽 (Banter):** AI 会对玩家与其他角色的对话产生“内心评论”。
    * **精准记忆过滤:** AI 在思考时会自动过滤掉无关的“噪音记忆”，使对话保持专注和逻辑性。

## 🧠 "AI灵魂"架构

本项目的核心是 `AI_Soul` 类，它封装了 AI 的所有行为：

1.  **RAG 检索 (Retrieval):** 当 AI 接收到刺激（如玩家对话），它会查询 `chromadb` 中的“事件”和“反思”两个记忆库。
2.  **相关性过滤:** 系统会自动过滤掉与当前情景无关的“噪音记忆”（基于向量距离阈值）。
3.  **Prompt 注入:** 将“基础人设”、“当前内部状态”（如情绪）、“相关记忆”组合成一个复杂的 Prompt。
4.  **LLM 生成 (CoT):** `ollama` 生成包含 `[Internal Monologue]` (内心独白) 和 `[Spoken Response]` (口头回复) 的回复。
5.  **反思闭环:** 系统捕获 `[Internal Monologue]` 并将其作为一条新记忆存回“反思记忆库”。
6.  **副作用 (Side Effects):** 同时，系统会触发好感度评估（对玩家或对同伴）、将互动存为“事件记忆”等。

## 🔬 功能验证 (Validation Highlights)

原型已成功验证 AI 灵魂架构的复杂涌现能力：

* **验证基础交互与好感度**:
    我们首先验证了AI的基础响应。例如，通过与角色“Falcon”进行不同对话，原型能准确展示其（基于人设的）**对玩家好感度的动态变化**。

* **验证C2C互动闭环**:
    我们验证了AI间的自主对话。当两个AI交流时，原型能完整记录**双方好感度的相互影响**，并将该次互动**作为新记忆分别存入各自的数据库**。

* **验证“个性+记忆”的复杂涌现**:
    我们测试了一个复杂情景：让“傲娇”的Mia被“冷静”的Ryan拯救。在注入这个“事件记忆”后，Mia对Ryan的看法发生了改变——她的想法**结合了“傲娇”人设、战场经历以及“被拯救”记忆**，表现出“（想感激但嘴上不服）那个碍眼的家伙…我连一句谢谢都没有说”的复杂情感。

* **验证“反思闭环”与情感深化**:
    在上述测试后，我们能立即在Mia的“反思记忆库”中看到她新产生的“（...明明是个废物，偏偏在战场上救了我）”的内心独白，以及**由该反思进一步触发的**后续自责：“(我连自己都保护不了…)”。这证明了AI不仅在记忆，还在**基于记忆进行多轮反思**。

* **验证“相关性过滤”**:
    我们对Falcon同时注入了“黑石行动”（重要事件）和“喜欢苹果”（日常偏好）的记忆。当与Falcon谈论“黑石行动”时，系统日志显示“喜欢苹果”的记忆**被成功过滤**，没有污染Falcon的回复；反之亦然。这保证了AI对话的专注和逻辑性。

## 本地部署 (Local Setup)

1.  **克隆仓库:**
    ```bash
    git clone https://github.com/ziyanglichuan/AI-soul.git
    cd AI-Soul
    ```

2.  **安装 Ollama:**
    请访问 [ollama.com](https://ollama.com/) 并根据您的操作系统（Windows, macOS, Linux）安装 Ollama。

3.  **拉取 LLM 模型:**
    原型演示使用的是 `qwen3:14b`（您也可以在代码中修改为其他模型，如 `llama3`）。
    ```bash
    ollama pull qwen3:14b
    ```

4.  **下载 AIGC 图片生成模型:**
    本项目使用 `cagliostrolab/animagine-xl-3.0` 进行图片生成。您需要从 Hugging Face Hub 下载此模型，并将其放置到项目中。
    * **模型链接:** [https://huggingface.co/cagliostrolab/animagine-xl-3.0](https://huggingface.co/cagliostrolab/animagine-xl-3.0)
    * **推荐做法:** `diffusers` 库会在首次运行时自动下载并缓存模型。您也可以手动下载模型文件并指定本地路径。

5.  **安装 Python 依赖:**
    建议使用虚拟环境。
    ```bash
    pip install -r requirements.txt
    ```

6.  **运行原型:**
    ```bash
    streamlit run web_demo.py
    ```
