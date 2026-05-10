import hashlib
import os
import re
import time
from collections import Counter
from pathlib import Path

import requests
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from openai import OpenAI
from playsound import playsound

OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "http://localhost:5000/v1")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "lm-studio")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "qwen2/world2")

CAPSWRITER_MARKDOWN_PATH = os.getenv("CAPSWRITER_MARKDOWN_PATH", "")
CHATTTS_AUDIO_FOLDER = os.getenv("CHATTTS_AUDIO_FOLDER", "")
CHATTTS_TTS_URL = os.getenv("CHATTTS_TTS_URL", "http://127.0.0.1:9966/tts")

BAIDU_TRANSLATE_APP_ID = os.getenv("BAIDU_TRANSLATE_APP_ID", "")
BAIDU_TRANSLATE_SECRET_KEY = os.getenv("BAIDU_TRANSLATE_SECRET_KEY", "")

ENTRY_PATTERN = re.compile(r"\[(\d{2}:\d{2}:\d{2})\]\((.+\.mp3)\)\s*(.*)")

client = OpenAI(base_url=OPENAI_BASE_URL, api_key=OPENAI_API_KEY)
sentiment_analyzer = SentimentIntensityAnalyzer()
chat_history = []


def translate_to_english(text: str) -> str:
    """Translate Chinese text to English for VADER sentiment analysis."""
    if not BAIDU_TRANSLATE_APP_ID or not BAIDU_TRANSLATE_SECRET_KEY:
        return text

    salt = str(time.time())
    sign = hashlib.md5(
        (BAIDU_TRANSLATE_APP_ID + text + salt + BAIDU_TRANSLATE_SECRET_KEY).encode("utf-8")
    ).hexdigest()
    params = {
        "q": text,
        "from": "zh",
        "to": "en",
        "appid": BAIDU_TRANSLATE_APP_ID,
        "salt": salt,
        "sign": sign,
    }
    response = requests.get("https://api.fanyi.baidu.com/api/trans/vip/translate", params=params, timeout=20)
    response.raise_for_status()
    data = response.json()
    if "trans_result" not in data:
        print(f"Baidu translate returned an unexpected response: {data}")
        return text
    return data["trans_result"][0]["dst"]


def chat_with_model(user_input: str) -> str:
    chat_history.append({"role": "user", "content": user_input})
    completion = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {
                "role": "system",
                "content": "你是一个乐于助人、聪明、善良、高效的人工智能助手，用简洁的话语回复。",
            },
            *chat_history,
        ],
        temperature=0.3,
    )
    response = completion.choices[0].message.content
    chat_history.append({"role": "assistant", "content": response})
    return response


def classify_sentiment(text: str) -> str:
    translated = translate_to_english(text)
    scores = sentiment_analyzer.polarity_scores(translated)
    compound = scores["compound"]
    print(f"Sentiment scores: {scores}")
    if compound >= 0.2:
        return "肯定"
    if compound <= -0.2:
        return "否定"
    return "中性"


def summarize_user_state(conversations: list[list[str]]) -> None:
    if not conversations:
        print("暂无足够对话数据。")
        return
    flat = [item for group in conversations for item in group]
    counts = Counter(flat)
    most_common = counts.most_common(1)[0][0]
    print("\n--- 对话状态摘要 ---")
    print(f"当前最常见状态：{most_common}")
    for name in ["肯定", "中性", "否定"]:
        print(f"{name}: {counts.get(name, 0)}")
    print("-------------------\n")


def read_latest_entry(markdown_path: Path) -> str | None:
    latest_match = None
    with markdown_path.open("r", encoding="utf-8", errors="ignore") as file:
        for line in file:
            match = ENTRY_PATTERN.search(line)
            if match:
                latest_match = match
    return latest_match.group(3).strip() if latest_match else None


def wait_for_new_description(markdown_path: Path, last_modified_time: float) -> tuple[str, float]:
    while True:
        current_modified_time = markdown_path.stat().st_mtime
        if current_modified_time != last_modified_time:
            description = read_latest_entry(markdown_path)
            if description:
                return description, current_modified_time
        time.sleep(2)


def get_latest_audio_file(folder: Path) -> Path | None:
    if not folder.exists():
        return None
    wav_files = list(folder.glob("*.wav"))
    if not wav_files:
        return None
    return max(wav_files, key=lambda item: item.stat().st_mtime)


def play_latest_audio() -> None:
    if not CHATTTS_AUDIO_FOLDER:
        return
    latest_file = get_latest_audio_file(Path(CHATTTS_AUDIO_FOLDER))
    if latest_file:
        print(f"Playing: {latest_file}")
        playsound(str(latest_file))


def speak(text: str) -> None:
    try:
        response = requests.post(
            CHATTTS_TTS_URL,
            data={
                "text": text,
                "prompt": "",
                "voice": "3333",
                "temperature": 0.3,
                "top_p": 0.7,
                "top_k": 20,
                "refine_max_new_token": "384",
                "infer_max_new_token": "2048",
                "skip_refine": 0,
                "is_split": 1,
                "custom_voice": 0,
            },
            timeout=60,
        )
        response.raise_for_status()
        print("TTS finished.")
        play_latest_audio()
    except Exception as exc:
        print(f"TTS skipped or failed: {exc}")


def main() -> None:
    if not CAPSWRITER_MARKDOWN_PATH:
        raise RuntimeError("请先设置 CAPSWRITER_MARKDOWN_PATH，指向 CapsWriter 生成的 Markdown 文件。")

    markdown_path = Path(CAPSWRITER_MARKDOWN_PATH)
    if not markdown_path.exists():
        raise FileNotFoundError(f"找不到语音转写文件: {markdown_path}")

    print("开始监听语音转写文件，输入 quit 退出，输入 end 重置当前对话。")
    last_modified_time = markdown_path.stat().st_mtime
    conversations: list[list[str]] = []
    temporary_states: list[str] = []

    while True:
        user_input, last_modified_time = wait_for_new_description(markdown_path, last_modified_time)
        print(f"收到新输入: {user_input}")

        command = user_input.lower().strip()
        if command == "quit":
            summarize_user_state(conversations)
            break
        if command == "end":
            summarize_user_state(conversations)
            conversations = []
            temporary_states = []
            chat_history.clear()
            continue

        attitude = classify_sentiment(user_input)
        temporary_states.append(attitude)
        if len(temporary_states) == 3:
            conversations.append(temporary_states)
            temporary_states = []
        if len(conversations) >= 2:
            summarize_user_state(conversations)
            conversations = []

        reply = chat_with_model(user_input)
        print("模型:", reply)
        speak(reply)


if __name__ == "__main__":
    main()
