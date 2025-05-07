import jieba
import requests
import json
import numpy as np
from hmmlearn import hmm
from openai import OpenAI
import requests
import os
import time
import re
from playsound import playsound
from nltk.sentiment.vader import SentimentIntensityAnalyzer  # 导入情感分析库
import hashlib

# 定义词汇表
vocabulary = ["肯定", "否定", "中性"]  # 您可以根据需要修改这个列表

# 设置 OpenAI 客户端
client = OpenAI(base_url="http://localhost:5000/v1", api_key="lm-studio")

chat_history = []  # 记录对话历史

def chat_with_model(user_input):
    """调用 AI 进行对话"""
    chat_history.append({"role": "user", "content": user_input})  # 记录用户输入
    completion = client.chat.completions.create(
        model="qwen2/world2",
        messages=[{"role": "system", "content": "你是一个乐于助人、聪明、善良、高效的人工智能助手，用简洁的话语回复。"}] + chat_history,
        temperature=0.3,
    )
    response = completion.choices[0].message.content
    chat_history.append({"role": "assistant", "content": response})  # 记录 AI 响应
    return response


# 分析用户状态的函数
def analyze_user_state(conversations):
    pass  # 如果函数还没有实现，可以暂时使用 pass


print("开始与模型对话，输入'quit'退出，输入'end'结束当前对话并开始新对话。")
###############################################################################
def analyze_user_state(all_conversations):
    if len(all_conversations) == 0:
        print("警告：没有观察到的对话数据。")
        return

    # 观测序列
    observations = all_conversations
    print(observations)
    # 将单词转换为数字编码的序列
    vocab2id = dict(zip(vocabulary, range(len(vocabulary))))

    def sentence2ids(sentence):
        return [vocab2id[word] for word in sentence]

    # 转换观测数据
    X = [sentence2ids(sentence) for sentence in observations]

    # 拼接成单一数组，并记录每个序列的长度
    lengths = [len(x) for x in X]
    data = np.concatenate(X)

    # 定义状态和单词
    states = ["认同", "怀疑", "否定"]
    id2topic = dict(zip(range(len(states)), states))

    # 初始概率
    start_probs = np.array([0.3, 0.4, 0.3])

    # 定义一个小的非零值
    SMALL_VALUE = 1e-10

    # 确保转移矩阵的每一行和为1
    trans_mat = np.array([[0.6, 0.3, 0.1],
                          [0.3, 0.4, 0.3],
                          [0.1, 0.3, 0.6]])

    # 每个状态下的发射概率（词汇的分布）
    emission_probs = np.array([[0.6, 0.2, 0.2],  # 发射概率矩阵
                               [0.3, 0.4, 0.3],
                               [0.2, 0.2, 0.6]])

    # 设置模型时，明确指定所有参数
    model = hmm.CategoricalHMM(n_components=len(states), n_iter=1000, init_params='')  # 禁用 's', 't', 'e'
    model.startprob_ = start_probs
    model.transmat_ = trans_mat
    model.emissionprob_ = emission_probs
    print(trans_mat, emission_probs)
    # 训练模型
    model.fit(data.reshape(-1, 1), lengths)

    # 如果训练后转移矩阵有零值，手动应用平滑
    SMALL_VALUE = 1e-10
    if np.any(model.transmat_ == 0):
        print("警告：训练后转移矩阵为零，正在应用平滑...")
        model.transmat_ = np.maximum(model.transmat_, SMALL_VALUE)
        model.transmat_ = model.transmat_ / model.transmat_.sum(axis=1, keepdims=True)

    '''''
    # 如果训练后转移矩阵为零，手动重置
    if np.all(model.transmat_ == 0):
        print("警告：训练后转移矩阵为零，正在重置...")
        model.transmat_ = trans_mat
    '''''
    # 解码以找出最可能的状态序列
    logprob, received = model.decode(data.reshape(-1, 1), lengths)

    # 计算后验概率
    posterior_probs = np.zeros(len(states))
    for i, state in enumerate(received):
        emission_prob = model.emissionprob_[state, data[i]]
        if i > 0:
            previous_state = received[i - 1]
            transition_prob = model.transmat_[previous_state, state]
        else:
            transition_prob = 1.0
        posterior_probs[state] += np.log(emission_prob * transition_prob)

    posterior_probs = np.exp(posterior_probs - np.max(posterior_probs))
    posterior_probs /= posterior_probs.sum()

    most_probable_state_bayes = id2topic[np.argmax(posterior_probs)]

    print(f"\n--- 对话分析结果 ---")
    print(f"用户最可能的状态是：{most_probable_state_bayes}")
    print(f"各状态概率：")
    for state, prob in zip(states, posterior_probs):
        print(f"{state}: {prob:.2f}")
    print("-----------------------------\n")
###############################################################################
#语言输入
# 文件地址
file_path = r'F:\program_learn\project\CapsWriter-Offline\CapsWriter-Offline-Windows-64bit\2025\02\07.md'

# 定义正则表达式，用于匹配音频文件的链接
pattern = re.compile(r'\[(\d{2}:\d{2}:\d{2})\]\((.+\.mp3)\)\s*(.*)')

# 获取文件的初始修改时间
last_modified_time = os.path.getmtime(file_path)

def get_latest_entry():
    """读取文件并返回最新的匹配行"""
    latest_match = None
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            match = pattern.search(line)
            if match:
                latest_match = match  # 更新最新的匹配
    return latest_match

print("开始监控文件变化...")

###############################################################################
#音频
# 设定音频文件夹路径
audio_folder = r"F:\program_learn\project\ChatTTS-UI-0.84\static\wavs"

def get_latest_audio_file(folder):
    """获取文件夹中最新的音频文件"""
    files = [f for f in os.listdir(folder) if f.endswith('.wav')]
    if not files:
        return None
    # 获取最新的文件
    latest_file = max(files, key=lambda f: os.path.getmtime(os.path.join(folder, f)))
    return os.path.join(folder, latest_file)

def play_latest_audio(folder):
    latest_file = get_latest_audio_file(folder)
    if latest_file:
        print(f"Playing: {latest_file}")
        playsound(latest_file)  # 播放最新的音频文件
    else:
        print("No audio files found.")
###############################################################################
# 百度翻译API配置信息
APP_ID = '20231101001866725'
SECRET_KEY = 'Xc0L0ggYIGgj7M7N9UGq'
 
# 定义百度翻译函数
def translate_to_chinese(text):
    url = "http://api.fanyi.baidu.com/api/trans/vip/translate"
    salt = str(time.time())
    sign = hashlib.md5((APP_ID + text + salt + SECRET_KEY).encode('utf-8')).hexdigest()
    params = {
        'q': text,
        'from': 'zh',
        'to': 'en',
        'appid': APP_ID,
        'salt': salt,
        'sign': sign
    }
    response = requests.get(url, params=params)
    result = response.json()
 
    # 添加错误处理和日志记录
    if 'trans_result' in result:
        return result['trans_result'][0]['dst']
    else:
        # 打印错误信息和完整的API响应
        print(f"翻译API响应错误: {result}")
        return text  # 如果翻译失败，返回原文
###############################################################################
# 对话
conversation_count = 0
all_conversations = []
temp_conversation = []  # 临时存储每三次情感分析的数组

def wait_for_new_description():
    global last_modified_time
    while True:
        current_modified_time = os.path.getmtime(file_path)
        if current_modified_time != last_modified_time:
            last_modified_time = current_modified_time
            latest_match = get_latest_entry()
            if latest_match:
                return latest_match.group(3)
        time.sleep(2)


print("等待新的输入...")
description = wait_for_new_description()
print(f"收到新输入: {description}")

while True:
    user_input = description

    # 如果用户输入 'quit'，退出循环
    if user_input.lower() == "quit":
        print("退出聊天...")
        break

    if user_input.lower() == "quit":
        if all_conversations:
            analyze_user_state(all_conversations)
        break

    if user_input.lower() == "end":
        if all_conversations:
            analyze_user_state(all_conversations)
        all_conversations = []
        conversation_count = 0
        print("等待新的输入...")
        description = wait_for_new_description()
        print(f"收到新输入: {description}")
        continue

    # EN_input = PyDeepLX.translate(user_input)
    # EN_input = translate_to_english(user_input)
    EN_input = translate_to_chinese(user_input)
    print("英文：",EN_input)
    sid = SentimentIntensityAnalyzer()  # 初始化 SentimentIntensityAnalyzer 实例
    # 进行情感分析并打印结果
    sentiment_scores = sid.polarity_scores(EN_input)
    print(f"情感分析结果: {sentiment_scores}")  # 打印情感分析结果
    print(sentiment_scores['compound'])
    # 设置阈值
    positive_threshold = 0.2
    negative_threshold = -0.2
    attitude = None
    # 最终情感判断
    if sentiment_scores['compound'] >= positive_threshold:
        attitude = "肯定"
        print("肯定")  # 正面情感
    elif sentiment_scores['compound'] <= negative_threshold:
        attitude = "否定"
        print("否定")  # 负面情感
    else:
        attitude = "中性"
        print("中性")  # 中性情感


    model_reply = chat_with_model(user_input)
    print("模型: ", model_reply)


    temp_conversation.append(attitude)
    # 每三个情感分析结果组成一个数组，并放入 all_conversations
    if len(temp_conversation) == 3:
        all_conversations.append(temp_conversation)
        temp_conversation = []  # 重置临时数组
        conversation_count += 1

    if conversation_count == 2:
        analyze_user_state(all_conversations)
        all_conversations = []
        conversation_count = 0

    # 语言反馈API调用代码
    res = requests.post('http://127.0.0.1:9966/tts', data={
    "text": model_reply,
    "prompt": "",
    "voice": "3333",
    "temperature": 0.3,
    "top_p": 0.7,
    "top_k": 20,
    "refine_max_new_token": "384",
    "infer_max_new_token": "2048",
    "skip_refine": 0,
    "is_split": 1,
    "custom_voice": 0
    })
    print(res.json())
    print("转音完成")

    if __name__ == "__main__":
        play_latest_audio(audio_folder)
        # 播放完后继续执行其他代码
        print("Audio played, continuing with the rest of the program.")
        # 继续你想要的其他代码

    print("等待新的输入...")
    description = wait_for_new_description()
    print(f"收到新输入: {description}")

