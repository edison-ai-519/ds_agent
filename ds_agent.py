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

    # Peeling state (user-guided)
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


def build_user_guided_peel_prompt(state: AnalysisState, selected_part: str) -> List[Dict[str, str]]:
    system = {
        "role": "system",
        "content": (
            "你是一个进行“偏去”处理的分析智能体。"
            "用户已手动选择一个部件进行偏去。"
            "请仅执行这一次偏去，并评估核心特征/本质是否发生变化。"
            "输出必须为严格 JSON。"
        ),
    }

    comp_snapshot = [
        {"name": c.name, "functions": c.functions} for c in state.remaining_components
    ]

    user_payload = {
        "item": state.item,
        "selected_part": selected_part,
        "remaining_components": comp_snapshot,
        "remaining_features": state.remaining_features,
        "core_features_before": state.core_features,
        "instructions": (
            "对 selected_part 执行一次偏去，返回严格 JSON：\n"
            "{\n"
            "  \"selected_part\": \"string\",\n"
            "  \"peeled_part_features\": [\"string\", ...],\n"
            "  \"peeled_part_functions\": [\"string\", ...],\n"
            "  \"reason\": \"string\",\n"
            "  \"impact_on_system\": \"string\",\n"
            "  \"updated_state\": {\n"
            "    \"remaining_components\": [{\"name\": \"string\", \"functions\": [\"string\", ...]} , ...],\n"
            "    \"remaining_features\": [\"string\", ...],\n"
            "    \"core_features_after\": [\"string\", ...],\n"
            "    \"has_core_change\": true/false,\n"
            "    \"is_essence_changed\": true/false,\n"
            "    \"explanation\": \"string\"\n"
            "  }\n"
            "}\n"
            "注意：\n"
            "- has_core_change 表示核心特征集合是否发生变化。\n"
            "- is_essence_changed 仅在偏去后物体的最小本质已发生改变时为 true。\n"
            "- 请勿添加额外解释性文字。"
        ),
    }

    return [system, {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)}]


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
    state.essence_features = list(state.core_features)
    state.essence_reached = False
    return state


def perform_user_guided_peel(state: AnalysisState, selected_part: str) -> AnalysisState:
    messages = build_user_guided_peel_prompt(state, selected_part)
    content = chat_completion(messages)
    data = extract_json(content)
    if not data:
        print("偏去结果解析失败，模型返回如下：\n")
        print(content)
        raise ValueError("无法解析偏去结果的 JSON")

    selected = str(data.get("selected_part", "")).strip()
    peeled_part_features = [
        str(x).strip() for x in (data.get("peeled_part_features", []) or []) if str(x).strip()
    ]
    peeled_part_functions = [
        str(x).strip() for x in (data.get("peeled_part_functions", []) or []) if str(x).strip()
    ]
    reason = str(data.get("reason", "")).strip()
    impact = str(data.get("impact_on_system", "")).strip()

    updated = data.get("updated_state") or {}

    # Record peel step
    if selected:
        state.peel_steps.append(
            PeelStep(
                removed_part=selected,
                removed_functions=peeled_part_functions,
                reason=reason,
                impact_on_system=impact,
            )
        )

    # Update remaining components and features
    new_components: List[Component] = []
    for c in (updated.get("remaining_components", []) or []):
        name = str(c.get("name", "")).strip()
        functions = [str(f).strip() for f in (c.get("functions", []) or []) if str(f).strip()]
        if name:
            new_components.append(Component(name=name, functions=functions))
    state.remaining_components = new_components

    state.remaining_features = [
        str(x).strip() for x in (updated.get("remaining_features", []) or []) if str(x).strip()
    ]

    core_after = [str(x).strip() for x in (updated.get("core_features_after", []) or []) if str(x).strip()]
    has_core_change = bool(updated.get("has_core_change", False))
    is_essence_changed = bool(updated.get("is_essence_changed", False))

    if core_after:
        state.core_features = core_after
    if is_essence_changed:
        state.essence_reached = True
        state.essence_features = list(core_after) if core_after else list(state.core_features)

    return state


def render_summary(state: AnalysisState) -> str:
    lines: List[str] = []
    lines.append(f"物品：{state.item}")
    if state.core_features:
        lines.append("当前核心特征：" + ", ".join(state.core_features))
    if state.components:
        lines.append("初始部件→功能：")
        for c in state.components:
            func_str = ", ".join(c.functions) if c.functions else "-"
            lines.append(f"- {c.name} → {func_str}")
    if state.peel_steps:
        lines.append("已执行的偏去步骤（部件→影响功能）：")
        for i, step in enumerate(state.peel_steps, 1):
            lost = ", ".join(step.removed_functions) if step.removed_functions else "-"
            lines.append(f"{i}. 移除 {step.removed_part} → 影响功能：{lost}")
    if state.essence_reached:
        lines.append("提示：本质已发生改变，建议停止进一步偏去。")
        ef = ", ".join(state.essence_features) if state.essence_features else "(未明确)"
        lines.append(f"- 本质特征集合：{ef}")
    return "\n".join(lines)


# =============================
# Conversation / REPL
# =============================

def build_dialogue_system_context(state: AnalysisState) -> str:
    return (
        "你是一名与用户对话的分析智能体。已完成对目标物体的结构化分析与若干偏去步骤。"
        "回答用户问题时请结合以下上下文，并保持简洁明确：\n"
        f"物品：{state.item}\n"
        f"当前核心特征：{', '.join(state.core_features)}\n"
        "初始部件→功能：\n" +
        "\n".join([f"- {c.name} → {', '.join(c.functions)}" for c in state.components]) +
        ("\n偏去步骤：\n" + "\n".join([
            f"{i+1}. 移除 {s.removed_part} → {', '.join(s.removed_functions)}" for i, s in enumerate(state.peel_steps)
        ]) if state.peel_steps else "") +
        ("\n已检测到本质改变：" + ", ".join(state.essence_features) if state.essence_reached else "")
    )


def dialogue_reply(state: AnalysisState, user_input: str, extra_history: Optional[List[Dict[str, str]]] = None) -> str:
    system_context = build_dialogue_system_context(state)
    messages: List[Dict[str, str]] = [{"role": "system", "content": system_context}]
    if extra_history:
        messages.extend(extra_history)
    messages.append({"role": "user", "content": user_input})
    return chat_completion(messages, temperature=0.5)


def list_remaining_components(state: AnalysisState) -> str:
    if not state.remaining_components:
        return "(无可偏去部件)"
    return "\n".join([f"- {c.name}（功能：{', '.join(c.functions) if c.functions else '-'}）" for c in state.remaining_components])


def repl():
    print("欢迎使用物体特征偏去分析智能体（手动偏去版，多轮对话）。输入 'exit' 退出。\n")
    while True:
        item = input("请输入想分析的物品（或输入 'exit' 退出）：").strip()
        if not item:
            print("请输入要分析的物品名称。\n")
            continue
        if item.lower() == "exit":
            print("已退出。")
            return

        print("\n[1/2] 正在进行初始结构化分析……\n")
        try:
            state = perform_initial_analysis(item)
        except Exception as e:
            print(f"初始分析失败：{e}\n")
            continue

        print("[2/2] 初始分析完成。以下为部件列表（可偏去）：\n")
        print(list_remaining_components(state) + "\n")
        print("进入多轮模式。指令说明：\n- 直接输入部件名称：对该部件执行一次偏去\n- 输入 'peel 部件名'：同上\n- 输入 'status'：查看当前摘要\n- 输入 'list'：查看可偏去部件\n- 输入 'new'：开始分析新的物品\n- 输入 'exit'：退出\n")

        history: List[Dict[str, str]] = []
        while True:
            user_q = input("你：").strip()
            if not user_q:
                continue
            lower = user_q.lower()
            if lower == "exit":
                print("已退出。")
                return
            if lower == "new":
                print("\n—— 重新开始新的物品分析 ——\n")
                break
            if lower == "status":
                print("\n" + render_summary(state) + "\n")
                continue
            if lower == "list":
                print("\n可偏去部件列表：\n" + list_remaining_components(state) + "\n")
                continue

            # Parse peel command
            selected_part = None
            if lower.startswith("peel "):
                selected_part = user_q[5:].strip()
            else:
                selected_part = user_q

            if not selected_part:
                print("请指定要偏去的部件名。\n")
                continue

            # Optional: warn if part not in remaining list
            remaining_names = {c.name for c in state.remaining_components}
            if selected_part not in remaining_names:
                print(f"提示：'{selected_part}' 未在当前可偏去部件列表中，仍尝试进行偏去。\n")

            try:
                state = perform_user_guided_peel(state, selected_part)
                print("\n已完成一次偏去。当前状态：\n")
                print(render_summary(state) + "\n")
                if state.essence_reached:
                    print("本质已发生改变。根据设定，建议停止进一步偏去。你可以输入 'new' 重新分析其他物品，或 'exit' 退出。\n")
                else:
                    print("可继续输入下一个要偏去的部件，或输入 'list' 查看部件列表。\n")
            except Exception as e:
                print(f"偏去失败：{e}\n")
                continue

            # Dialogue mode is always available
            # If the user wants to ask questions unrelated to peel, they can prefix with '? '
            if lower.startswith("? "):
                try:
                    q = user_q[2:].strip()
                    reply = dialogue_reply(state, q, history)
                    print("智能体：" + reply + "\n")
                    history.extend([
                        {"role": "user", "content": q},
                        {"role": "assistant", "content": reply},
                    ])
                except Exception as e:
                    print(f"对话失败：{e}\n")


if __name__ == "__main__":
    repl()







