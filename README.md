# Vocabulary Learning Assistant / 词小灶

面向江苏高三英语课堂的轻量词汇教学助手。教师可以一次导入一个单元的词汇，生成词根词族与易混词辨析、无提示挖词填空，并记录课堂薄弱点、导出复习资料。

## 核心功能

- 一次导入本单元全部词汇
- 词根词族整理与高考易混词辨析
- 按主题生成高考风格挖词填空和语法填空
- 单词强化卡：例句、高频搭配和助记
- 记录薄弱词与课堂提问
- 生成课堂薄弱点 PDF

## 快速开始

### 1. 准备环境

需要 Python 3.10 或更高版本。

```powershell
git clone https://github.com/Meowemeow/vocabulary-learning-assistant.git
cd vocabulary-learning-assistant
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. 配置模型

```powershell
Copy-Item .env.example .env
```

编辑 `.env`，填写 OpenAI 兼容接口：

```env
API_KEY=your-api-key
API_BASE_URL=https://api.openai.com/v1
MODEL=gpt-4o-mini
FALLBACK_MODEL=
HOST=127.0.0.1
PORT=5000
```

`.env` 已被 Git 忽略，请勿将真实 API Key 提交到仓库。

### 3. 启动

Windows 用户可直接双击：

```text
启动词小灶.bat
```

也可以在终端运行：

```powershell
python app.py
```

浏览器访问 `http://127.0.0.1:5000`。

## 项目结构

```text
vocabulary-learning-assistant/
├─ app.py                  # Flask 后端与模型调用
├─ templates/
│  └─ index.html           # 教师操作页面
├─ requirements.txt        # Python 依赖
├─ .env.example            # 配置示例，不含密钥
├─ .gitignore
├─ 启动词小灶.bat           # Windows 一键启动
└─ README.md
```

## 数据说明

- 单元词汇和薄弱词记录主要保存在当前浏览器本地。
- 生成的临时文件位于 `output/` 或 `tmp/`，不会提交到 Git。
- 本项目不包含任何真实 API Key。

## 使用范围

本项目用于高中英语词汇课堂辅助。模型生成内容应由教师审核后用于教学。

## 作者

昆山市高级中学 申珅
