from openai import OpenAI
import os

ds_api_key = os.getenv('DEEPSEEK_API_KEY')
client = OpenAI(api_key=ds_api_key,base_url='https://api.deepseek.com/v1')

def analyze(item_name):
    prompt = f"""
    你是一个专业的分析师。请对“{item_name}”进行详细分析，并严格按照以下5个部分给出结果：

    1. **物品特征**: 列出{item_name}的所有主要物理和功能特征。
    2. **核心特征**: 从上述特征中，提取出{item_name}最关键的3-5个核心特征。
    3. **同类物品**: 列举出3-5个与{item_name}具有相同核心特征的同类物品。
    4. **系统描述**: 详细描述{item_name}的系统构成，包括其主要部件及其结构关系。
    5. **同部件物品**: 列举出3-5个与{item_name}具有相同或类似部件的同类物品。

    请直接给出分析结果，不要包含任何额外的对话或解释性文字。
    """
    message = [
        {"role": "system", "content": "你是一个专业的物品分析智能体。"},
        {"role": "user", "content": prompt}]
    response = client.chat.completions.create(
        messages = message,
        model = "deepseek-chat"
     )
    analyze_result = response.choices[0].message.content
    return analyze_result

item = input("请输入您想分析的物品：")
if not item:
     print("输入物品后我才能够分析")
print("\n正在分析中...请稍等...\n")
result = analyze(item)
print("以下为分析内容：")
print(result)
print("分析完毕")







