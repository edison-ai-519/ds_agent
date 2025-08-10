import os
import sys
import json
import re
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from openai import OpenAI


# =============================
# Configuration & Client Setup
# =============================

ds_api_key = os.getenv("DEEPSEEK_API_KEY")
if not ds_api_key:
    print("未检测到环境变量 DEEPSEEK_API_KEY。请设置后重试。")
    sys.exit(1)

client = OpenAI(api_key=ds_api_key, base_url="https://api.deepseek.com/v1")


# =============================
# Data Structures
# =============================

@dataclass
class Component:
    name: str
    functions: List[str] = field(default_factory=list)

@dataclass
class PeelStep:
    removed_part: str
    removed_functions: List[str]
    reason: str
    impact_on_system: str

@dataclass
class AnalysisState:
    item: str
    features: List[str] = field(default_factory=list)
    core_features: List[str] = field(default_factory=list)
    components: List[Component] = field(default_factory=list)
    relationships: List[str] = field(default_factory=list)

    # Iterative peeling state
    remaining_components: List[Component] = field(default_factory=list)
    remaining_features: List[str] = field(default_factory=list)
    peel_steps: List[PeelStep] = field(default_factory=list)
    essence_features: List[str] = field(default_factory=list)
    essence_reached: bool = False


# =============================
# LLM Helpers
# =============================

def chat_completion(messages: List[Dict[str, str]], temperature: float = 0.2) -> str:
    response = client.chat.completions.create(
        messages=messages,
        model="deepseek-chat",
        temperature=temperature,
    )
    return response.choices[0].message.content


def extract_json(text: str) -> Optional[Dict[str, Any]]:
    # Try to extract the first top-level JSON object from text
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Fallback: try to locate a JSON block heuristically
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        candidate = match.group(0)
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            return None
    return None


# =============================
# Prompts
# =============================

def build_initial_analysis_prompt(item_name: str) -> List[Dict[str, str]]:
    system = {
        "role": "system",
        "content": (
            "你是一个专业的物品分析与本质提炼智能体。"
            "你的输出必须严格遵循用户要求的 JSON 结构，且不要输出任何多余文本。"
        ),
    }
    user = {
        "role": "user",
        "content": f"""
请对“{item_name}”进行结构化分析，并只以 JSON 返回，JSON 结构如下：
{{
  "item": "string",
  "features": ["string", ...],
  "core_features": ["string", ...],
  "components": [
    {{"name": "string", "functions": ["string", ...]}},
    ...
  ],
  "relationships": ["string", ...]
}}

要求：
- components 中每个部件必须给出该部件承担的功能 functions（至少1条）。
- features 与 core_features 为物理特征与功能特征的混合集合，但 core_features 是最关键的3-5条。
- 严格输出上述 JSON，不要包含任何注释或额外解释。
""",
    }
    return [system, user]


def build_peel_prompt(state: AnalysisState) -> List[Dict[str, str]]:
    system = {
        "role": "system",
        "content": (
            "你是一个进行“偏去”处理的分析智能体。"
            "目标：逐步剥离非本质部件，建立“移除部件→失去/改变的功能”的对应关系，直到只剩本质特征。"
            "输出必须为严格 JSON。"
        ),
    }
    # Prepare compact state snapshot
    comp_snapshot = [
        {"name": c.name, "functions": c.functions} for c in state.remaining_components
    ]
    user_content = {
        "item": state.item,
        "remaining_components": comp_snapshot,
        "remaining_features": state.remaining_features,
        "core_features": state.core_features,
        "instructions": (
            "请执行一次偏去（如可能），返回严格 JSON：\n"
            "{\n"
            "  \"peel_step\": {\n"
            "    \"removed_part\": \"string\",\n"
            "    \"removed_functions\": [\"string\", ...],\n"
            "    \"reason\": \"string\",\n"
            "    \"impact_on_system\": \"string\"\n"
            "  },\n"
            "  \"updated_state\": {\n"
            "    \"remaining_components\": [{\"name\": \"string\", \"functions\": [\"string\", ...]} , ...],\n"
            "    \"remaining_features\": [\"string\", ...],\n"
            "    \"is_essence_reached\": true/false,\n"
            "    \"essence_features\": [\"string\", ...]\n"
            "  }\n"
            "}\n"
            "若已经无法继续偏去而不伤及本质，请将 is_essence_reached 设为 true，并给出 essence_features。"
        ),
    }
    return [system, {"role": "user", "content": json.dumps(user_content, ensure_ascii=False)}]


# =============================
# Core Logic
# =============================

def perform_initial_analysis(item_name: str) -> AnalysisState:
    messages = build_initial_analysis_prompt(item_name)
    content = chat_completion(messages)
    data = extract_json(content)
    if not data:
        print("初始分析解析失败，模型返回如下：\n")
        print(content)
        raise ValueError("无法解析初始分析的 JSON")

    components: List[Component] = []
    for c in data.get("components", []) or []:
        name = c.get("name", "").strip()
        functions = [str(f).strip() for f in (c.get("functions", []) or []) if str(f).strip()]
        if name:
            components.append(Component(name=name, functions=functions))

    state = AnalysisState(
        item=data.get("item", item_name),
        features=[str(x).strip() for x in (data.get("features", []) or []) if str(x).strip()],
        core_features=[str(x).strip() for x in (data.get("core_features", []) or []) if str(x).strip()],
        components=components,
        relationships=[str(x).strip() for x in (data.get("relationships", []) or []) if str(x).strip()],
    )

    # Initialize remaining state
    state.remaining_components = [Component(name=c.name, functions=list(c.functions)) for c in components]
    state.remaining_features = list(state.features)
    return state


def perform_iterative_peeling(state: AnalysisState, max_steps: int = 20) -> AnalysisState:
    steps = 0
    while steps < max_steps and not state.essence_reached:
        messages = build_peel_prompt(state)
        content = chat_completion(messages)
        data = extract_json(content)
        if not data:
            # If parsing fails, stop to avoid infinite loop
            print("偏去结果解析失败，模型返回如下：\n")
            print(content)
            break

        peel = data.get("peel_step") or {}
        updated = data.get("updated_state") or {}

        removed_part = str(peel.get("removed_part", "")).strip()
        removed_functions = [str(x).strip() for x in (peel.get("removed_functions", []) or []) if str(x).strip()]
        reason = str(peel.get("reason", "")).strip()
        impact = str(peel.get("impact_on_system", "")).strip()

        if removed_part:
            state.peel_steps.append(
                PeelStep(
                    removed_part=removed_part,
                    removed_functions=removed_functions,
                    reason=reason,
                    impact_on_system=impact,
                )
            )

        # Update remaining components & features
        new_components: List[Component] = []
        for c in (updated.get("remaining_components", []) or []):
            name = c.get("name", "").strip()
            functions = [str(f).strip() for f in (c.get("functions", []) or []) if str(f).strip()]
            if name:
                new_components.append(Component(name=name, functions=functions))
        state.remaining_components = new_components

        state.remaining_features = [
            str(x).strip() for x in (updated.get("remaining_features", []) or []) if str(x).strip()
        ]

        state.essence_reached = bool(updated.get("is_essence_reached", False))
        state.essence_features = [
            str(x).strip() for x in (updated.get("essence_features", []) or []) if str(x).strip()
        ]

        steps += 1

    return state


def render_summary(state: AnalysisState) -> str:
    lines: List[str] = []
    lines.append(f"物品：{state.item}")
    if state.core_features:
        lines.append("核心特征：" + ", ".join(state.core_features))
    if state.components:
        lines.append("初始部件→功能：")
        for c in state.components:
            func_str = ", ".join(c.functions) if c.functions else "-"
            lines.append(f"- {c.name} → {func_str}")
    if state.peel_steps:
        lines.append("偏去过程（部件→失去/改变的功能）：")
        for i, step in enumerate(state.peel_steps, 1):
            lost = ", ".join(step.removed_functions) if step.removed_functions else "-"
            lines.append(f"{i}. 移除 {step.removed_part} → 影响功能：{lost}")
            if step.reason:
                lines.append(f"   理由：{step.reason}")
            if step.impact_on_system:
                lines.append(f"   系统影响：{step.impact_on_system}")
    if state.essence_reached:
        lines.append("已到达本质：")
        ef = ", ".join(state.essence_features) if state.essence_features else "(模型未明确给出)"
        lines.append(f"- 本质特征集合：{ef}")
        lines.append("建议：停止偏去")
    else:
        lines.append("尚未确认到达本质（或达到最大步骤限制）。")
    return "\n".join(lines)


# =============================
# Conversation / REPL
# =============================

def build_dialogue_system_context(state: AnalysisState) -> str:
    return (
        "你是一名与用户对话的分析智能体。已完成对目标物体的结构化分析与偏去处理。"
        "回答用户问题时请结合以下上下文，并保持简洁明确：\n" 
        f"物品：{state.item}\n"
        f"核心特征：{', '.join(state.core_features)}\n"
        "初始部件→功能：\n" +
        "\n".join([f"- {c.name} → {', '.join(c.functions)}" for c in state.components]) +
        ("\n偏去步骤：\n" + "\n".join([
            f"{i+1}. 移除 {s.removed_part} → {', '.join(s.removed_functions)}" for i, s in enumerate(state.peel_steps)
        ]) if state.peel_steps else "") +
        ("\n已到达本质：" + ", ".join(state.essence_features) if state.essence_reached else "")
    )


def dialogue_reply(state: AnalysisState, user_input: str, extra_history: Optional[List[Dict[str, str]]] = None) -> str:
    system_context = build_dialogue_system_context(state)
    messages: List[Dict[str, str]] = [{"role": "system", "content": system_context}]
    if extra_history:
        messages.extend(extra_history)
    messages.append({"role": "user", "content": user_input})
    return chat_completion(messages, temperature=0.5)


def repl():
    print("欢迎使用物体特征偏去分析智能体（多轮版）。输入 'exit' 退出。\n")
    while True:
        item = input("请输入想分析的物品（或输入 'exit' 退出）：").strip()
        if not item:
            print("请输入要分析的物品名称。\n")
            continue
        if item.lower() == "exit":
            print("已退出。")
            return

        print("\n[1/3] 正在进行初始结构化分析……\n")
        try:
            state = perform_initial_analysis(item)
        except Exception as e:
            print(f"初始分析失败：{e}\n")
            continue

        print("[2/3] 正在进行偏去处理（自动迭代直到本质或达到上限）……\n")
        state = perform_iterative_peeling(state)

        print("[3/3] 结果汇总：\n")
        print(render_summary(state))
        print("\n进入多轮对话模式。你可以：\n- 直接就该物品或偏去结果发问\n- 输入 'status' 查看当前摘要\n- 输入 'new' 重新分析其他物品\n- 输入 'exit' 退出\n")

        history: List[Dict[str, str]] = []
        while True:
            user_q = input("你：").strip()
            if not user_q:
                continue
            if user_q.lower() == "exit":
                print("已退出。")
                return
            if user_q.lower() == "new":
                print("\n—— 重新开始新的物品分析 ——\n")
                break
            if user_q.lower() == "status":
                print("\n" + render_summary(state) + "\n")
                continue

            try:
                reply = dialogue_reply(state, user_q, history)
                print("智能体：" + reply + "\n")
                history.extend([
                    {"role": "user", "content": user_q},
                    {"role": "assistant", "content": reply},
                ])
            except Exception as e:
                print(f"对话失败：{e}\n")


if __name__ == "__main__":
    repl()







