import ollama
import chromadb
import time
import json
import re
import torch 
from diffusers import ( 
    StableDiffusionXLPipeline, 
    EulerAncestralDiscreteScheduler,
    AutoencoderKL
)
from PIL import Image 
from typing import List, Optional # 导入类型提示

# AI_Soul
class AI_Soul:
    def __init__(self, soul_id: str, base_persona: str, model: str):
        self.soul_id = soul_id
        self.base_persona = base_persona 
        self.model = model 
        
        self.internal_state = {
            "mood": "Neutral",      
            "status": "Healthy",   
        }
        self.favorability_player = 0 
        self.favorability_peers = {} 
        
        self.db_client = chromadb.PersistentClient(path=f"./soul_db_{soul_id}")
        self.event_memory = self.db_client.get_or_create_collection(name="events")
        self.reflection_memory = self.db_client.get_or_create_collection(name="reflections")
        
        self.SIMILARITY_THRESHOLD = 0.1 # 用于去重

        # 只有距离小于 1.2 的记忆才被认为 "相关"
        self.RETRIEVAL_THRESHOLD = 1.2  
        
        print(f"[System]: AI_Soul '{self.soul_id}' initialized. State: {self.internal_state}")

    def set_persona(self, new_persona: str):
        self.base_persona = new_persona
        print(f"[System]: Persona for '{self.soul_id}' updated.")

    def clear_all_memories(self):
        print(f"[System]: Clearing all memories for '{self.soul_id}'...")
        try:
            self.db_client.delete_collection(name="events")
            self.db_client.delete_collection(name="reflections")
        except Exception:
            pass 
        self.event_memory = self.db_client.get_or_create_collection(name="events")
        self.reflection_memory = self.db_client.get_or_create_collection(name="reflections")

    def set_state(self, key: str, value):
        if key in self.internal_state:
            self.internal_state[key] = value
        else:
            print(f"[State Error]: Unknown state key '{key}'")
            
    def add_peer(self, peer_id: str):
        if peer_id not in self.favorability_peers:
            self.favorability_peers[peer_id] = 0

    def _add_memory_if_unique(self, collection, doc_to_add: str, metadata: dict, id_prefix: str):
        try:
            results = collection.query(
                query_texts=[doc_to_add],
                n_results=1,
                include=["distances"] 
            )
            
            if results['distances'] and results['distances'][0]:
                distance = results['distances'][0][0]
                if distance < self.SIMILARITY_THRESHOLD:
                    print(f"\n[Memory Skipped (Duplicate)]: Distance {distance:.4f} < {self.SIMILARITY_THRESHOLD}")
                    return False
            
            mem_id = f"{id_prefix}_{int(time.time())}"
            collection.add(
                documents=[doc_to_add],
                ids=[mem_id],
                metadatas=[metadata]
            )
            print(f"\n[Memory Injected ({metadata['type']} @ {metadata.get('interlocutor', 'N/A')})]: '{doc_to_add}'")
            return True

        except Exception as e:
            print(f"[Error in _add_memory_if_unique]: {e}")
            return False

    # --- MODIFIED ---
    def add_event_memory(self, fact: str, importance: int = 5, interlocutor: str = "System"):
        """
        为记忆添加 'interlocutor' (对话者) 元数据。
        'System' 表示一般性事实, 'Player' 表示玩家, 其他 soul_id 表示NPC。
        """
        metadata = {
            "importance": importance, 
            "type": "event",
            "interlocutor": interlocutor  # <-- 新增的元数据
        }
        self._add_memory_if_unique(
            collection=self.event_memory, 
            doc_to_add=fact, 
            metadata=metadata, 
            id_prefix="evt"
        )

    def add_reflection_memory(self, reflection: str, importance: int = 3):
        # 反思没有 interlocutor，因为它们是内部的
        metadata = {"importance": importance, "type": "reflection"}
        self._add_memory_if_unique(
            collection=self.reflection_memory, 
            doc_to_add=reflection, 
            metadata=metadata, 
            id_prefix="ref"
        )


    # --- MODIFIED ---
    # 支持相关性过滤 和 按对话者过滤
    def retrieve_memories(self, query: str, n_results: int = 2, filter_by_interlocutors: Optional[List[str]] = None) -> list:
        if not query:
            query = "general thoughts" 
            
        # 1. 构建元数据过滤器
        # 总是检索 "System" (通用事实)
        interlocutor_list = ["System"]
        if filter_by_interlocutors:
            interlocutor_list.extend(filter_by_interlocutors)
            
        # 过滤器: (interlocutor IN ["System", "Player", "Peer_ID", ...])
        event_where_filter = {
            "interlocutor": {
                "$in": interlocutor_list
            }
        }
            
        try:
            # 查询事件 (带过滤)
            print(f"[Memory Retrieve]: Filtering events using: {event_where_filter}")
            events_results = self.event_memory.query(
                query_texts=[query], 
                n_results=n_results, 
                where=event_where_filter, # <--- 应用过滤器
                include=["documents", "distances"]
            )
            
            # 查询反思 (不需要过滤，反思总是相关的)
            reflections_results = self.reflection_memory.query(
                query_texts=[query], 
                n_results=n_results, 
                include=["documents", "distances"]
            )
        except Exception as e:
            print(f"[Memory Retrieve Error]: {e}")
            return []

        retrieved_with_dist = []
        
        # 过滤事件
        if events_results['documents'] and events_results['documents'][0]:
            for doc, dist in zip(events_results['documents'][0], events_results['distances'][0]):
                if dist < self.RETRIEVAL_THRESHOLD:
                    retrieved_with_dist.append((f"[Event]: {doc}", dist))
                else:
                    print(f"[Memory Filtered (Irrelevant)]: Event '{doc[:30]}...' (Dist: {dist:.4f} >= {self.RETRIEVAL_THRESHOLD})")

        # 过滤反思
        if reflections_results['documents'] and reflections_results['documents'][0]:
            for doc, dist in zip(reflections_results['documents'][0], reflections_results['distances'][0]):
                if dist < self.RETRIEVAL_THRESHOLD:
                    retrieved_with_dist.append((f"[Reflection]: {doc}", dist))
                else:
                    print(f"[Memory Filtered (Irrelevant)]: Reflection '{doc[:30]}...' (Dist: {dist:.4f} >= {self.RETRIEVAL_THRESHOLD})")
        
        # 按相关性对记忆进行排序
        retrieved_with_dist.sort(key=lambda x: x[1])
        
        # 返回排序最相关的 top-n 个记忆
        final_docs = [doc for doc, dist in retrieved_with_dist][:n_results]

        if final_docs:
            print(f"[Memories Retrieved (Relevant)]: {json.dumps(final_docs, ensure_ascii=False)}")
        else:
            print("[Memories Retrieved]: No *relevant* memories found.")
            
        return final_docs

    def _parse_cot_response(self, full_response: str) -> (str, str):
        monologue = "..."
        response = "..." 
        mono_match = re.search(r"\[(?:Internal Monologue|内心评论|内心独白)\]:\s*(.*?)(?=\[(?:Spoken Response|口头回复)\]:|$)", full_response, re.DOTALL)
        if mono_match:
            monologue = mono_match.group(1).strip()
            
        spoken_match = re.search(r"\[(?:Spoken Response|口头回复|开场白)\]:\s*(.*)", full_response, re.DOTALL)
        if spoken_match:
            response = spoken_match.group(1).strip()
            
        if not spoken_match and mono_match:
             response = mono_match.group(1).strip() 

        if not mono_match and not spoken_match and full_response:
             response = full_response.strip()

        return monologue, response

    def _get_favorability_change(self, interaction_prompt: str) -> int:
        meta_prompt = f"""
        [你的情景]: 你的人设是 '{self.base_persona}'
        [刚发生的互动]:
        {interaction_prompt}
        [你的任务]:
        请基于你的“核心人设”评估这次互动。这次互动让你的好感度增加了多少或减少了多少？
        请在-5（非常讨厌）到+5（非常喜欢）的范围内给出一个整数值。
        [你的回复]:
        请只回复一个数字（例如：-2, 0, 3）。
        数字:
        """
        try:
            print(f"[V8.1 Favorability]: 正在评估互动: {interaction_prompt[:50]}...")
            response = ollama.chat(
                model=self.model,
                messages=[{'role': 'user', 'content': meta_prompt}],
                options={'temperature': 0.0}
            )
            change_str = response['message']['content'].strip()
            change_num_str = re.findall(r"(-?\d+)", change_str)
            if change_num_str:
                change = int(change_num_str[0])
                print(f"[V8.1 Favorability]: 评估结果: {change}")
                return max(-5, min(5, change))
            else:
                print(f"[V8.1 Favorability]: 解析数字失败，回复: '{change_str}'")
                return 0 
        except Exception as e:
            print(f"[V8.1 Favorability Error]: LLM 评估失败: {e}")
            return 0
            
    def update_peer_favorability(self, peer_id: str, interaction: str):
        if peer_id not in self.favorability_peers:
            self.add_peer(peer_id)
        change = self._get_favorability_change(f"你和 {peer_id} 刚刚的互动:\n{interaction}")
        self.favorability_peers[peer_id] += change
        print(f"[V8.1 Favorability]: {self.soul_id} 对 {peer_id} 的好感度变为 {self.favorability_peers[peer_id]} (变化: {change})")
        return change

    # --- MODIFIED ---
    def _generate_core_response(self, query_for_rag: str, full_llm_prompt_template: str, filter_interlocutors: Optional[List[str]] = None) -> (str, str, list):
        
        # 检索记忆 (传入过滤器)
        relevant_memories = self.retrieve_memories(
            query_for_rag, 
            filter_by_interlocutors=filter_interlocutors # <--- 传递过滤器
        ) 
        memories_str = "\n".join(relevant_memories) if relevant_memories else "无"
        
        # 将记忆注入到 Prompt 模板中
        try:
            final_prompt = full_llm_prompt_template.format(
                memories=memories_str,
                state=json.dumps(self.internal_state, ensure_ascii=False)
            )
        except KeyError as e:
            print(f"[Prompt Template Error]: Prompt 模板缺少 {e} 占位符。")
            final_prompt = full_llm_prompt_template # 降级处理
        
        try:
            response = ollama.chat(
                model=self.model,
                messages=[{'role': 'user', 'content': final_prompt}],
                options={'temperature': 0.5} 
            )
            full_text = response['message']['content']
            
            monologue, spoken_response = self._parse_cot_response(full_text)
            
            if monologue and monologue != "...":
                # 反思是内部的，不需要 interlocutor
                self.add_reflection_memory(monologue, importance=1)
                
            return monologue, spoken_response, relevant_memories
            
        except Exception as e:
            print(f"[Ollama Error in Core Response]: {e}")
            return "...", f"SYSTEM ERROR: {e}", relevant_memories


    # --- MODIFIED ---
    def generate_response_to_player(self, query: str):
        
        prompt_template = f"""
你正在扮演一个虚拟战士。
[你的核心人设]: {self.base_persona} 
[你的当前内部状态]: {{state}}
[你脑海中闪过的相关记忆]: {{memories}}
[情景]: 'Player' 刚刚对你说了话。
[任务]:
1. [Internal Monologue]: 写下你的“真实内心独白”。
2. [Spoken Response]: 决定你“选择”对外说什么。
User: "{query}"
---
[Internal Monologue]:
[Spoken Response]:
"""
        
        monologue, spoken_response, memories = self._generate_core_response(
            query_for_rag=query, 
            full_llm_prompt_template=prompt_template,
            filter_interlocutors=["Player"]  # <--- 告诉 RAG 只检索和玩家相关的
        )
            
        # [新增] 将与玩家的互动本身记录为“事件”
        interaction_fact = f"玩家对我说: '{query}', 我回应: '{spoken_response}'"
        self.add_event_memory(interaction_fact, importance=4, interlocutor="Player") # <--- 明确指定 interlocutor

        # 评估对玩家的好感度
        favor_change = self._get_favorability_change(f"玩家对你说了: '{query}'")
        self.favorability_player += favor_change
        print(f"[Favorability]: {self.soul_id} 对玩家的好感度变为 {self.favorability_player} (变化: {favor_change})")
        
        return monologue, spoken_response, memories, favor_change
        
    # --- MODIFIED ---
    def generate_system_response(self, query_for_rag: str, full_llm_prompt_template: str, filter_interlocutors: Optional[List[str]] = None) -> (str, str, list):
        monologue, spoken_response, memories = self._generate_core_response(
            query_for_rag=query_for_rag, 
            full_llm_prompt_template=full_llm_prompt_template,
            filter_interlocutors=filter_interlocutors # <--- 传递 C2C/Banter 过滤器
        )
        return monologue, spoken_response, memories


    def _get_persona_keywords(self) -> str:
        print(f"[V8.0 Keyword Gen]: 正在调用 LLM ({self.model}) 总结人设...")
        meta_prompt = f"""
        你是一个专业的提示词工程师。
        请将以下复杂的“角色人设”总结为 5-10 个简短的、逗号分隔的英文“视觉关键词”，
        用于喂给AI绘图模型(如 Stable Diffusion)。
        请只关注外表、装备、气质和关键特征。
        [人设描述]: "{self.base_persona}"
        [你的任务]: 请只输出这些关键词，用逗号分隔，不要说任何其他的话。
        例如: "1girl, captain, proud, battle-worn, scar, red armor, sci-fi"
        """
        try:
            response = ollama.chat(
                model=self.model, 
                messages=[{'role': 'user', 'content': meta_prompt}],
                options={'temperature': 0.2} 
            )
            keywords = response['message']['content'].strip()
            print(f"[Keyword Gen]: LLM 总结的关键词: {keywords}")
            return keywords
        except Exception as e:
            print(f"[Keyword Gen Error]: LLM 总结失败: {e}")
            return "warrior, character" 

    def get_image_generation_prompt(self) -> str:
        persona_keywords = self._get_persona_keywords()
        image_prompt = (
            f"High-quality collectible action figure of one person, {persona_keywords}, "
            f"style: anime figurine, character illustration, dynamic pose, studio lighting, "
            f"realistic materials, displayed on a small base, full body shot."
        )
        print(f"[Image Prompt]: {image_prompt}")
        return image_prompt


# --- MODIFIED ---
def trigger_soul_banter(
    bystander_soul: AI_Soul, 
    active_soul_id: str, 
    user_prompt: str, 
    ai_response: str
) -> str:
    print(f"[Banter System]: 触发 {bystander_soul.soul_id} 的评论...")
    
    # 1. 定义用于RAG的查询
    query_for_rag = f"对 {active_soul_id} 的看法, 以及关于 '{user_prompt}' 的对话"
    
    # 2. 定义Prompt模板 (包含 {memories} 和 {state})
    banter_prompt_template = f"""
[你的核心人设]: {bystander_soul.base_persona}
[你的当前内部状态]: {{state}}
[你脑海中闪过的相关记忆]: {{memories}}

[你的情景]:
你正在“旁观”玩家与 '{active_soul_id}' 的对话。
[你刚听到的对话]:
玩家: "{user_prompt}"
{active_soul_id}: "{ai_response}"

[你的任务]:
请基于你的“核心人设”和你的“相关记忆”，对这场对话发表一句简短的、一针见血的“内心评论”。
请只使用 [Internal Monologue]: 标签输出你的想法。

[Internal Monologue]:
"""
    
    print(f"[Banter System]: {bystander_soul.soul_id} 正在(带记忆)生成评论...")
    
    monologue, spoken, _ = bystander_soul.generate_system_response(
        query_for_rag, 
        banter_prompt_template,
        # 旁观者会检索关于玩家和关于被评论者的记忆
        filter_interlocutors=[active_soul_id, "Player"] 
    )
    
    comment = spoken if spoken and spoken != "..." else monologue
    
    if comment and comment != "...":
        print(f"[Banter System]: {bystander_soul.soul_id} 评论: {comment}")
        return comment
    return ""


# --- MODIFIED ---
def trigger_soul_c2c_dialogue(soul_a: AI_Soul, soul_b: AI_Soul):
    print(f"[C2C System]: {soul_a.soul_id} 决定是否对 {soul_b.soul_id} 发起对话。")
    
    query_for_rag_a = f"我对 {soul_b.soul_id} ({soul_b.base_persona}) 的看法和过往"
    
    spark_prompt_template = f"""
[你的核心人设]: {soul_a.soul_id} ({soul_a.base_persona})
[你的当前内部状态]: {{state}}
[你脑海中闪过的相关记忆]: {{memories}}

[你的情景]:
你现在正和 {soul_b.soul_id} ({soul_b.base_persona}) 待在一起。
你决定（基于你的记忆和人设）主动对 {soul_b.soul_id} 开启一个话题。

[你的任务]:
请生成你的“内心独白”和你要说的“开场白”（一句简短的话）。

[Internal Monologue]:
[Spoken Response]:
"""
    # A 只检索关于 B 的记忆
    mono_a, line_a, _ = soul_a.generate_system_response(
        query_for_rag_a, 
        spark_prompt_template,
        filter_interlocutors=[soul_b.soul_id] 
    )
    
    if (not line_a or line_a == "...") and (mono_a and mono_a != "..."):
        line_a = mono_a 
    
    if not line_a or line_a == "...":
        print(f"[C2C System]: {soul_a.soul_id} 决定保持沉默。")
        return None

    query_for_rag_b = f"{soul_a.soul_id} 对我说了 '{line_a}'"
    
    response_prompt_template = f"""
[你的核心人设]: {soul_b.soul_id} ({soul_b.base_persona})
[你的当前内部状态]: {{state}}
[你脑海中闪过的相关记忆]: {{memories}}

[你的情景]:
{soul_a.soul_id} ({soul_a.base_persona}) 刚刚对你说了: "{line_a}"

[你的任务]:
请基于你的“核心人设”和“相关记忆”，生成你的“内心独白”和对 {soul_a.soul_id} 的“口头回复”。

[Internal Monologue]:
[Spoken Response]:
"""
    # B 只检索关于 A 的记忆
    mono_b, line_b, _ = soul_b.generate_system_response(
        query_for_rag_b, 
        response_prompt_template,
        filter_interlocutors=[soul_a.soul_id]
    )

    if (not line_b or line_b == "...") and (mono_b and mono_b != "..."):
        line_b = mono_b 

    if not line_b or line_b == "...":
        print(f"[C2C System]: {soul_b.soul_id} 决定不回应。")
        line_b = f"*(...{soul_b.soul_id} 沉默地看了看 {soul_a.soul_id})*" 

    interaction_a_to_b = f"我主动对 {soul_b.soul_id} 说：'{line_a}'，{soul_b.soul_id}回应：'{line_b}'"
    interaction_b_to_a = f"{soul_a.soul_id} 对我说：'{line_a}'，我回应{soul_b.soul_id}：'{line_b}'"
    
    change_a = soul_a.update_peer_favorability(soul_b.soul_id, interaction_a_to_b)
    change_b = soul_b.update_peer_favorability(soul_a.soul_id, interaction_b_to_a)
    
    # 存储记忆时，明确指定对话者
    soul_a.add_event_memory(interaction_a_to_b, importance=5, interlocutor=soul_b.soul_id) # <--- A 存储 B
    soul_b.add_event_memory(interaction_b_to_a, importance=5, interlocutor=soul_a.soul_id) # <--- B 存储 A
    
    print(f"[C2C System] Log: {soul_a.soul_id} -> {soul_b.soul_id}: {line_a}")
    print(f"[C2C System] Log: {soul_b.soul_id} (回复): {line_b}")
    print(f"[C2C System] Log: (好感度: {soul_a.soul_id}❤️{soul_b.soul_id} {change_a:+} | {soul_b.soul_id}❤️{soul_a.soul_id} {change_b:+} )")

    return (soul_a.soul_id, soul_b.soul_id, line_a, line_b, change_a, change_b)