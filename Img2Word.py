# test.py
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer

# 初始化模型和分词器
model = AutoModel.from_pretrained(
    'openbmb/MiniCPM-V-2_6-int4',
    trust_remote_code=True,
    load_in_4bit=False,  # 显式关闭4bit加载
    load_in_8bit=False   # 显式关闭8bit加载
    )
tokenizer = AutoTokenizer.from_pretrained('openbmb/MiniCPM-V-2_6-int4', trust_remote_code=True)
model.eval()

# 加载作文图片
image = Image.open('img.png').convert('RGB')

# 设置明确的OCR指令
ocr_prompt = "请直接提取图片中的全部文字，不要添加任何解释、格式或标点修正，直接输出原始文字内容。最后给出评分"

# 构建消息格式
msgs = [{'role': 'user', 'content': [image, ocr_prompt]}]

# 进行文字识别
result = model.chat(
    image=image,  # 直接传入图片对象
    msgs=msgs,
    tokenizer=tokenizer,
    sampling=False,  # 关闭采样确保最大确定性
    temperature = 0.7
)

# 保存结果到txt文件
with open('output.txt', 'w', encoding='utf-8') as f:
    f.write(result)

print("文字已成功保存到output.txt")