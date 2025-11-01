import streamlit as st
from ai_soul_core import AI_Soul, trigger_soul_banter, trigger_soul_c2c_dialogue
import time
import json 
import os 
import random

import torch
from diffusers import (
    StableDiffusionXLPipeline, 
    EulerAncestralDiscreteScheduler,
    AutoencoderKL
)
from PIL import Image 

st.set_page_config(
    page_title="AI-Soul",
    page_icon="🤖",
    layout="wide" 
)

# 定义模型常量
LLM_MODEL = "qwen3:14b" 
IMAGE_GPU_DEVICE = torch.device("cuda:1") 
LOCAL_IMAGE_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "animagine-xl-3.0")
NEGATIVE_PROMPT = "nsfw, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, artist name"

# 内置角色模板
CHARACTER_TEMPLATES = {
    "alex": {
        "name": "Alex",
        "persona": "一名代号'战神'的突击队长。极度骄傲、好战，十分看重实力。坚信进攻是最好的防守。",
        "memory": "在'黑石'行动中，因为轻敌冒进，导致小队陷入埋伏，失去了最好的搭档。"
    },
    "ryan": {
        "name": "Ryan",
        "persona": "一名代号'荒野'的精英狙击手。冷静、言简意赅，从不让情绪干扰判断。只相信自己的瞄准镜。",
        "memory": "在一次雪地任务中，为了掩护撤退，独自一人牵制了敌方一个排，三天后才归队。"
    },
    "mia": {
        "name": "Mia",
        "persona": "一名代号'幽魂'的女性技术专家和无人机操作员。性格傲娇，喜欢独处，总是在和她的机器低语。",
        "memory": "曾在一次边境冲突中，独自潜入敌方营地侦查，为己方提供关键情报，但目睹战友在前线牺牲。"
    }
}


@st.cache_resource
def load_sdxl_pipeline():
    device = IMAGE_GPU_DEVICE
    print(f"\n[SDXL Model Loader]: 目标设备: {device}")
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        print("[SDXL Model Loader Error]: 未找到 GPU 1 或 CUDA 不可用。")
        st.error(f"错误：未找到 GPU 1。")
        return None 
    MODEL_PATH = LOCAL_IMAGE_MODEL_PATH
    VAE_PATH = os.path.join(MODEL_PATH, "vae")
    if not (os.path.isdir(MODEL_PATH) and os.path.isdir(VAE_PATH)):
        print(f"[SDXL Model Loader Error]: 找不到模型目录 '{MODEL_PATH}' 或 'vae' 子目录。")
        st.error(f"错误：找不到模型目录 '{MODEL_PATH}' 或 'vae' 子目录。")
        return None
    print(f"[SDXL Model Loader]: 正在加载 VAE (从本地 {VAE_PATH})...")
    vae = AutoencoderKL.from_pretrained(VAE_PATH, torch_dtype=torch.float16)
    vae.to(device)
    print(f"[SDXL Model Loader]: VAE 加载完成。")
    print(f"[SDXL Model Loader]: 正在加载 StableDiffusionXLPipeline (从本地 {MODEL_PATH})...")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        MODEL_PATH, vae=vae, torch_dtype=torch.float16, use_safetensors=True, 
    )
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe.to(device) 
    print(f"[SDXL Model Loader]: StableDiffusionXLPipeline 加载完成并移至 {device}。")
    return pipe

if 'app_state' not in st.session_state:
    st.session_state.app_state = "creation"  
if 'characters' not in st.session_state:
    st.session_state.characters = {}  
if 'character_images' not in st.session_state:
    st.session_state.character_images = {}  
if 'chat_histories' not in st.session_state:
    st.session_state.chat_histories = {}  
if 'active_soul_id' not in st.session_state:
    st.session_state.active_soul_id = None  
if 'interaction_log' not in st.session_state: 
    st.session_state.interaction_log = [] 

def render_creation_view():
    st.title("角色创建工坊")
    st.caption("您可以选择预设模板，也可以完全自定义您的角色。最多创建 3 名虚拟战士。")

    st.subheader("我的手办架")
    cols_shelf = st.columns(3)
    created_char_ids = list(st.session_state.characters.keys())
    for i in range(3):
        with cols_shelf[i]:
            with st.container(border=True): 
                if i < len(created_char_ids):
                    soul_id = created_char_ids[i]
                    st.image(st.session_state.character_images[soul_id], caption=soul_id, width='stretch') 
                else:
                    st.markdown("*(空位)*")
                    st.container(height=200)
    
    st.divider()

    if len(st.session_state.characters) < 3:
        st.subheader("从模板创建角色")
        available_templates = {}
        for template_id, data in CHARACTER_TEMPLATES.items():
            if template_id not in st.session_state.characters:
                available_templates[template_id] = data
                
        if available_templates:
            cols_templates = st.columns(len(available_templates))
            for i, (template_id, data) in enumerate(available_templates.items()):
                with cols_templates[i]:
                    with st.container(border=True):
                        st.markdown(f"**{data['name']}**")
                        st.caption(f"_{data['persona'][:50]}..._") 
                        if st.button(f"✨ 实例化 {data['name']}", key=f"create_template_{template_id}", width='stretch'):
                            create_character(template_id, data['persona'], data['memory'])
        else:
            st.info("所有预设模板均已创建。")
            
        st.divider()

        st.subheader("或手动创建新角色")
        with st.form(key="custom_character_creation_form"):
            custom_char_name = st.text_input("角色名称 (必须唯一)", placeholder="例如：Falcon, Shadow...", max_chars=20)
            custom_char_persona = st.text_area("角色描述 (人设)", placeholder="例如：一名代号'幽影'的侦察兵，行动敏捷...", height=150)
            custom_char_memory = st.text_area("初始关键记忆 (可选)", placeholder="例如：在一次救援行动中被困三天...", height=100)
            
            submit_custom_button = st.form_submit_button("自定义创建并生成形象")

            if submit_custom_button:
                soul_id = custom_char_name.lower().replace(" ", "_")
                if not custom_char_name or not custom_char_persona:
                    st.error("角色名称和角色描述不能为空！")
                elif soul_id in st.session_state.characters:
                    st.error(f"角色名称 '{custom_char_name}' 已存在（或其ID '{soul_id}' 已被占用），请输入一个唯一的名称。")
                else:
                    create_character(soul_id, custom_char_persona, custom_char_memory)
    else:
        st.warning("角色栏已满（3/3）。请先进入手办柜体验，或刷新页面重置。")

    if st.session_state.characters:
        st.divider()
        if st.button("✅ 完成创建，进入手办柜", type="primary"):
            st.session_state.app_state = "chat"
            st.rerun()

def create_character(soul_id: str, persona: str, memory: str = ""):
    with st.spinner(f"正在创建 {soul_id}, 并生成形象..."): 
        new_soul = AI_Soul(soul_id, persona, LLM_MODEL)
        if memory:
            new_soul.add_event_memory(memory) 
            
        existing_souls = st.session_state.characters
        for existing_id, existing_soul in existing_souls.items():
            new_soul.add_peer(existing_id)      
            existing_soul.add_peer(soul_id)     
            print(f"建立关系 {soul_id} <-> {existing_id}")
            
        pipe_sdxl = load_sdxl_pipeline()
        if pipe_sdxl is None:
            st.error("图片模型加载失败，无法创建角色。请检查控制台。")
            st.stop()
            
        prompt = new_soul.get_image_generation_prompt()
        image = pipe_sdxl(
            prompt, negative_prompt=NEGATIVE_PROMPT, 
            width=832, height=1216, 
            guidance_scale=7, num_inference_steps=28
        ).images[0]
        
        st.session_state.characters[soul_id] = new_soul
        st.session_state.character_images[soul_id] = image
        st.session_state.chat_histories[soul_id] = []
        
        if st.session_state.active_soul_id is None:
            st.session_state.active_soul_id = soul_id
            
        st.success(f"角色 '{soul_id}' 创建成功！")
        time.sleep(1)
        st.rerun()

def render_chat_view():
    
    active_soul_id = st.session_state.active_soul_id
    if not active_soul_id:
        st.error("错误：没有活跃角色。请先创建角色。")
        if st.button("返回角色创建"):
            st.session_state.app_state = "creation"
            st.rerun()
        st.stop()
        
    active_soul = st.session_state.characters[active_soul_id]
    active_history = st.session_state.chat_histories[active_soul_id]

    main_chat_col, shelf_col = st.columns([2, 1]) 

    # 手办架 
    with shelf_col:
        st.subheader("手办架")
        
        shelf_slots = st.columns(3) 
        char_ids = list(st.session_state.characters.keys())
        
        for i in range(3): 
            with shelf_slots[i]: 
                with st.container(border=True):
                    if i < len(char_ids):
                        soul_id = char_ids[i]
                        soul = st.session_state.characters[soul_id]
                        
                        st.image(st.session_state.character_images[soul_id], caption=soul_id, width=130) 
                        
                        fav_info = f"""
                        <div style='
                            line-height: 1.2; 
                            margin-top: 0px; 
                            margin-bottom: 5px; 
                            text-align: center;
                        '>
                            <p style='margin: 0;'>❤️ 玩家: {soul.favorability_player}</p>
                        """
                        # 同伴好感度
                        peer_favs = []
                        for peer_id, score in soul.favorability_peers.items():
                            peer_favs.append(f"vs {peer_id}: {score}")

                        if peer_favs:
                            fav_info += f"<p style='font-size: 0.8em; margin: 0;'>{', '.join(peer_favs)}</p>"

                        fav_info += "</div>"

                        st.markdown(fav_info, unsafe_allow_html=True)
                        
                        if soul_id == active_soul_id:
                            st.button(f"**正在对话**", disabled=True, width='stretch', type='primary')
                        else:
                            if st.button(f"与 {soul_id} 对话", key=f"switch_{soul_id}", width='stretch'):
                                st.session_state.active_soul_id = soul_id
                                st.rerun()
                    else:
                        st.markdown(f"*(空位)*")
                        st.container(height=100)
        
        st.markdown("<div style='margin-top: 5px; margin-bottom: 5px;'></div>", unsafe_allow_html=True) 
        
        st.subheader("角色动态 (Interaction Log)") 
        with st.container(height=197, border=True): 
            if not st.session_state.interaction_log:
                st.caption("*(角色间暂无互动)*")
            for interaction in reversed(st.session_state.interaction_log): 
                st.markdown(interaction)
        
        if st.button("返回角色创建", key="go_to_creation"):
            st.session_state.app_state = "creation"
            st.rerun()

    # 主聊天窗口
    with main_chat_col:
        st.title(f"正在与 {active_soul_id} 对话")
        st.caption(f"人设: {active_soul.base_persona[:100]}...") 
        
        with st.container(height=400, border=True):
            for message in active_history:
                if message["role"] == "user":
                    with st.chat_message("user", avatar="🧑‍💻"):
                        st.markdown(message["content"])
                elif message["role"] == "assistant":
                    with st.chat_message("assistant", avatar="🤖"):
                        with st.expander("🧠 **AI思考中 (点击展开)**"):
                            if "memories" in message and message["memories"]: 
                                st.caption("检索到的记忆：")
                                for mem in message["memories"]:
                                    display_memory_card(mem) 
                            else:
                                st.caption("未检索到相关记忆。")
                            st.info(f"**内心独白:** {message['monologue']}")
                        st.markdown(message["spoken"])
                elif message["role"] == "system":
                    with st.chat_message("system", avatar="⚙️"):
                        st.markdown(f"*{message['content']}*")
        
        inject_col, clear_col = st.columns([3, 1])

        with inject_col:
            with st.form(key="event_injection_form", clear_on_submit=True):
                event_text = st.text_input(
                    f"向 {active_soul_id} 注入新记忆：", 
                    placeholder="例如：在黑石行动中险胜",
                    label_visibility="collapsed"
                )
                inject_button = st.form_submit_button(f"💉 注入事件记忆", use_container_width=True)

                if inject_button and event_text:
                    with st.spinner(f"正在向 {active_soul_id} 注入记忆..."):
                        active_soul.add_event_memory(event_text, importance=10) 
                    
                    active_history.append({
                        "role": "system",
                        "content": f"[系统事件]: 记忆“{event_text}”已注入。"
                    })
                    st.toast(f"记忆已注入 {active_soul_id}！", icon="💉")
                    st.rerun() 
        
        with clear_col:
            with st.expander(f"🗑️ 清空 {active_soul_id} 记忆", expanded=False):
                if st.button("确认清空", use_container_width=True, type="primary"):
                    with st.spinner(f"正在清空 {active_soul_id} 的所有记忆..."):
                        active_soul.clear_all_memories() 
                    
                    active_history.append({
                        "role": "system",
                        "content": f"[系统事件]: {active_soul_id} 的所有记忆已被清空。"
                    })
                    st.toast(f"{active_soul_id} 的记忆已清空！", icon="🗑️")
                    st.rerun()


        if prompt := st.chat_input(f"与 {active_soul_id} 对话..."):
            
            active_history.append({"role": "user", "content": prompt})
            
            with st.spinner(f"{active_soul_id} 正在思考... AI们可能也在交谈..."):
                
                mono, spoken, memories, favor_change = active_soul.generate_response_to_player(prompt)
                
                active_history.append({
                    "role": "assistant",
                    "monologue": mono,
                    "spoken": spoken,
                    "memories": memories 
                })
                
                if favor_change > 0:
                    st.toast(f"{active_soul_id} ❤️ 好感度 +{favor_change}!", icon="❤️")
                elif favor_change < 0:
                    st.toast(f"{active_soul_id} 💔 好感度 {favor_change}!", icon="💔")

                if len(st.session_state.characters) > 1:
                    r = random.random()
                    if r < 0.3:
                        bystander_ids = [sid for sid in st.session_state.characters if sid != active_soul_id]
                        if bystander_ids:
                            bystander_id = random.choice(bystander_ids)
                            bystander_soul = st.session_state.characters[bystander_id]
                            
                            comment = trigger_soul_banter(bystander_soul, active_soul_id, prompt, spoken)
                            
                            if comment:
                                new_log = f"**{bystander_id}** (评论): {comment}"
                                st.session_state.interaction_log.append(new_log)
                                if len(st.session_state.interaction_log) > 5:
                                    st.session_state.interaction_log.pop(0)

                    elif r < 0.8:
                        all_soul_ids = list(st.session_state.characters.keys())
                        if len(all_soul_ids) >= 2:
                            soul_a_id, soul_b_id = random.sample(all_soul_ids, 2)
                            soul_a = st.session_state.characters[soul_a_id]
                            soul_b = st.session_state.characters[soul_b_id]
                            
                            result = trigger_soul_c2c_dialogue(soul_a, soul_b)
                            
                            if result:
                                a_id, b_id, line_a, line_b, change_a, change_b = result
                                log_line_1 = f"**{a_id}** -> **{b_id}**: {line_a}"
                                log_line_2 = f"**{b_id}** (回复): {line_b}"
                                log_line_3 = f"*(好感度: {a_id}❤️{b_id} {change_a:+} | {b_id}❤️{a_id} {change_b:+} )*"
                                
                                st.session_state.interaction_log.append(log_line_1)
                                st.session_state.interaction_log.append(log_line_2)
                                st.session_state.interaction_log.append(log_line_3) 
                                
                                while len(st.session_state.interaction_log) > 6: 
                                    st.session_state.interaction_log.pop(0)
            
            st.rerun() 


def display_memory_card(mem: str):
    st.markdown(
        f"""
        <div style="
            background-color: #f0f2f6; 
            border-radius: 5px; 
            padding: 10px; 
            margin: 5px 0; 
            color: #31333F; 
            font-family: 'Source Sans Pro', sans-serif;
            border: 1px solid #ddd;
        ">
            {mem}
        </div>
        """,
        unsafe_allow_html=True
    )


if st.session_state.app_state == "creation":
    render_creation_view()
elif st.session_state.app_state == "chat":
    render_chat_view()
else:
    st.error("应用状态错误")
    st.session_state.app_state = "creation"
    st.rerun()