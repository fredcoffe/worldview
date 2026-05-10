# Worldview

这是一个语音输入驱动的本地 AI 对话实验项目。它监听 CapsWriter 生成的 Markdown 转写文件，读取最新一句话，交给本地或 OpenAI 兼容模型回复，再用 ChatTTS 接口转成语音播放。

## 它做什么

1. 监听语音转写 Markdown 文件。
2. 读取最新输入文本。
3. 可选调用百度翻译，把中文转成英文做情感分析。
4. 调用 OpenAI 兼容接口生成回复。
5. 调用本地 ChatTTS 接口生成语音。
6. 简单统计用户对话状态：肯定、中性、否定。

## 快速运行

```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt
copy .env.example .env
python test3_final.py
```

macOS/Linux 激活虚拟环境：

```bash
source .venv/bin/activate
```

## 必要配置

复制 `.env.example` 为 `.env`，填入本机路径和接口地址。默认模型接口是 LM Studio：`http://localhost:5000/v1`。

## 安全说明

- 百度翻译 `APP_ID` 和 `SECRET_KEY` 已改为环境变量读取。
- 旧版本里曾出现明文百度翻译密钥，公开仓库后建议立刻去百度平台重置。
- 不要提交 `.env`、本地音频、转写文件或任何真实 API Key。
