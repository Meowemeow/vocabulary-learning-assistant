from __future__ import annotations

import os
import json
import io
import re
import sys
import threading
import webbrowser
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from flask import Flask, Response, jsonify, render_template, request, send_file, stream_with_context
from openai import OpenAI
import httpx2

BUNDLE_DIR = Path(__file__).resolve().parent
BASE_DIR = Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else BUNDLE_DIR
TEMPLATE_DIR = BUNDLE_DIR / "templates"
load_dotenv(BASE_DIR / ".env", override=True)
app = Flask(__name__, template_folder=str(TEMPLATE_DIR))
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024

SYSTEM_PROMPT = """你是“词小灶”，服务于江苏高三英语教师的课堂词汇教学。
你只处理高中英语词汇学习、词根词族、易混词辨析、语境填空、例句、搭配、助记和薄弱词复习。
不回答任何课程无关问题，不进行无关闲聊，不扩展与任务无关的内容。
课堂对象是一个班级，不是一对一长期陪练。
输出内容要符合高中英语和江苏高考语境，语言准确、简洁、可直接用于课堂。
"""


def get_client() -> OpenAI:
    # 每次请求重新读取 .env，教师修改模型后无需重启服务。
    load_dotenv(BASE_DIR / ".env", override=True)
    api_key = (os.getenv("NANOBOT_API_KEY", "").strip() or os.getenv("API_KEY", "").strip())
    if not api_key:
        raise RuntimeError("尚未配置 API_KEY，请在 .env 文件中填写 API Key。")

    kwargs: dict[str, Any] = {"api_key": api_key}
    base_url = (os.getenv("NANOBOT_API_URL", "").strip() or os.getenv("API_BASE_URL", "").strip())
    if base_url:
        kwargs["base_url"] = base_url
    # 不使用系统中可能残留的代理环境变量，避免请求被转发到失效的本地代理。
    http_client = httpx2.Client(trust_env=False, timeout=180.0)
    return OpenAI(**kwargs, timeout=180.0, max_retries=1, http_client=http_client)


def create_model_stream(user_prompt: str, max_tokens: int = 6000) -> tuple[OpenAI, Any, str, str]:
    client = get_client()
    primary_model = (os.getenv("NANOBOT_MODEL", "").strip() or os.getenv("MODEL", "gpt-4o-mini").strip())
    fallback_model = os.getenv("FALLBACK_MODEL", "").strip()

    def request_stream(model: str):
        return client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            stream=True,
            # DeepSeek 官方接口：普通课堂任务关闭深度思考，减少延迟和格式漂移。
            extra_body={"thinking": {"type": "disabled"}},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )

    try:
        return client, request_stream(primary_model), primary_model, primary_model
    except Exception as primary_exc:
        status_code = int(getattr(primary_exc, "status_code", 0) or 0)
        can_fallback = bool(fallback_model and fallback_model != primary_model and status_code >= 500)
        if not can_fallback:
            client.close()
            raise

        app.logger.warning(
            "primary model %s failed with %s; falling back to %s",
            primary_model,
            status_code,
            fallback_model,
        )
        try:
            return client, request_stream(fallback_model), primary_model, fallback_model
        except Exception:
            client.close()
            raise

def collect_json_model(user_prompt: str, max_tokens: int = 8000, attempts: int = 2) -> str:
    """Call DeepSeek JSON Output mode and return validated, canonical JSON text."""
    last_error: Exception | None = None
    for attempt in range(attempts):
        client = get_client()
        primary_model = (os.getenv("NANOBOT_MODEL", "").strip() or os.getenv("MODEL", "deepseek-flash").strip())
        fallback_model = os.getenv("FALLBACK_MODEL", "").strip()
        models = [primary_model]
        if fallback_model and fallback_model != primary_model:
            models.append(fallback_model)
        prompt = user_prompt
        if attempt:
            prompt += "\n\n上一次返回未能形成完整JSON。本次请压缩表述，严格输出一个完整、合法且闭合的JSON对象。"
        try:
            for model in models:
                try:
                    response = client.chat.completions.create(
                        model=model,
                        max_tokens=max_tokens,
                        stream=False,
                        response_format={"type": "json_object"},
                        extra_body={"thinking": {"type": "disabled"}},
                        messages=[
                            {"role": "system", "content": SYSTEM_PROMPT + "\n你必须严格输出合法JSON对象。"},
                            {"role": "user", "content": prompt},
                        ],
                    )
                    choice = response.choices[0]
                    finish_reason = str(getattr(choice, "finish_reason", "") or "")
                    if finish_reason == "length":
                        raise RuntimeError("模型输出达到长度上限，JSON未完整返回。")
                    content = (choice.message.content or "").strip()
                    if not content:
                        raise RuntimeError("模型返回了空JSON内容。")
                    parsed = json.loads(content)
                    if not isinstance(parsed, dict):
                        raise RuntimeError("模型返回的顶层结构不是JSON对象。")
                    return json.dumps(parsed, ensure_ascii=False, separators=(",", ":"))
                except Exception as exc:
                    last_error = exc
                    status_code = int(getattr(exc, "status_code", 0) or 0)
                    if model == primary_model and len(models) > 1 and status_code >= 500:
                        continue
                    break
        finally:
            client.close()
    raise RuntimeError(f"模型未能返回完整JSON：{last_error}")


def model_json_response(user_prompt: str, max_tokens: int = 8000) -> Response:
    """Return validated JSON through the existing SSE envelope used by the webpage."""
    @stream_with_context
    def generate():
        try:
            content = collect_json_model(user_prompt, max_tokens=max_tokens)
            yield sse_message("delta", content=content)
            yield sse_message("done")
        except Exception as exc:
            app.logger.exception("model json failed")
            yield sse_message("error", error=f"调用模型失败：{exc}")

    return Response(
        generate(),
        content_type="text/event-stream; charset=utf-8",
        headers={"Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no"},
    )


def collect_model_text(user_prompt: str, max_tokens: int = 6000) -> str:
    client, stream, _primary_model, _used_model = create_model_stream(user_prompt, max_tokens=max_tokens)
    pieces: list[str] = []
    try:
        for chunk in stream:
            if not chunk.choices:
                continue
            content = getattr(chunk.choices[0].delta, "content", None)
            if content:
                pieces.append(content)
    finally:
        close_stream = getattr(stream, "close", None)
        if callable(close_stream):
            close_stream()
        client.close()
    result = "".join(pieces).strip()
    if not result:
        raise RuntimeError("模型没有返回可用内容。")
    return result


def sse_message(event_type: str, **payload: Any) -> str:
    return f"data: {json.dumps({'type': event_type, **payload}, ensure_ascii=False)}\n\n"


def model_stream_response(user_prompt: str, max_tokens: int = 6000) -> Response:
    client, stream, primary_model, used_model = create_model_stream(user_prompt, max_tokens=max_tokens)

    @stream_with_context
    def generate():
        has_content = False
        try:
            if used_model != primary_model:
                yield sse_message(
                    "notice",
                    notice=f"主模型 {primary_model} 暂时不可用，已自动切换备用模型 {used_model}。",
                    model=used_model,
                )
            for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                content = getattr(delta, "content", None)
                if content:
                    has_content = True
                    yield sse_message("delta", content=content)
            if not has_content:
                yield sse_message("error", error="模型没有返回可显示的内容。")
            else:
                yield sse_message("done")
        except Exception as exc:
            app.logger.exception("model stream failed")
            yield sse_message("error", error=f"调用模型失败：{exc}")
        finally:
            close_stream = getattr(stream, "close", None)
            if callable(close_stream):
                close_stream()
            client.close()

    return Response(
        generate(),
        content_type="text/event-stream; charset=utf-8",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
        },
    )


def read_json() -> dict[str, Any]:
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        raise ValueError("请求内容格式不正确。")
    return data


def require_text(data: dict[str, Any], name: str) -> str:
    value = str(data.get(name, "")).strip()
    if not value:
        raise ValueError(f"缺少必要内容：{name}")
    return value


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/health")
def health():
    configured = bool((os.getenv("NANOBOT_API_KEY", "").strip() or os.getenv("API_KEY", "").strip()))
    return jsonify({"ok": True, "configured": configured})


@app.post("/api/assistant")
def assistant():
    try:
        data = read_json()
        question = require_text(data, "question")
        prompt = f"""【课堂小助手】

教师问题：
{question}

请作为江苏高三英语课堂小助手回答问题。
要求：
1. 只回答高中英语词汇、语法、阅读、写作、单元教学和本智能体操作相关问题；
2. 如果问题与课程无关，只回复：“我只回答高中英语学习和本课堂智能体相关问题。”；
3. 回答简短、直接，适合教师课堂即时使用；
4. 一般控制在300字以内，必要时使用项目符号；
5. 不闲聊，不扩展，不反问，不输出与问题无关的内容。"""
        return model_stream_response(prompt, max_tokens=800)
    except (ValueError, RuntimeError) as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    except Exception as exc:
        app.logger.exception("assistant failed")
        return jsonify({"ok": False, "error": f"调用模型失败：{exc}"}), 502


@app.post("/api/roots")
def roots():
    try:
        data = read_json()
        vocabulary = require_text(data, "vocabulary")
        prompt = f"""【功能一：词根词族与江苏高考易混词辨析】

本单元全部词汇：
{vocabulary}

请对本单元词汇进行整体整理，并严格返回一个合法 JSON 对象。
不要使用 Markdown 代码块，不要在 JSON 前后添加说明，不要输出注释。

固定结构如下，字段名必须完全一致：
{{
  "title": "本单元词根词族与易混词辨析",
  "families": [
    {{
      "root": "词根、词缀或核心词",
      "rootMeaning": "核心含义",
      "words": [
        {{
          "word": "单词",
          "partOfSpeech": "词性",
          "meaning": "核心中文含义",
          "relation": "与本词族的构词或词义关系"
        }}
      ]
    }}
  ],
  "confusions": [
    {{
      "words": ["易混词1", "易混词2"],
      "difference": "一句话说明核心区别",
      "examTip": "简短的高考用法或搭配提示"
    }}
  ],
  "otherWords": [
    {{
      "word": "单词",
      "partOfSpeech": "词性",
      "meaning": "核心中文含义"
    }}
  ]
}}

要求：
1. 按明显、可靠的词根、词缀、派生或词形变化关系分组；
2. 每个词族单独作为 families 中的一项，同一个单词不要重复；
3. 没有可靠词族关系的单词放入 otherWords，不要牵强解释；
4. confusions 只选择本单元最值得辨析的词形、词义或词性易混词，最多10组；
5. 只处理输入词汇，不补充清单外的单词；
6. 内容简洁、准确，适合江苏高三英语课堂；
7. 所有顶层字段必须存在，没有内容时使用空数组；
8. 最终只输出可被 JSON.parse 直接解析的 JSON。"""
        return model_json_response(prompt, max_tokens=8000)
    except (ValueError, RuntimeError) as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    except Exception as exc:
        app.logger.exception("roots failed")
        return jsonify({"ok": False, "error": f"调用模型失败：{exc}"}), 502


@app.post("/api/fill")
def fill():
    try:
        data = read_json()
        vocabulary = require_text(data, "vocabulary")
        topic = require_text(data, "topic")
        prompt = f"""【功能二：挖词填空】

本单元全部词汇：
{vocabulary}

指定主题：{topic}

请生成课堂练习，并严格返回一个合法 JSON 对象。
不要使用 Markdown 代码块，不要在 JSON 前后添加说明，不要输出注释。

固定结构如下，字段名必须完全一致：
{{
  "title": "主题词汇语境训练",
  "readings": [
    {{
      "title": "阅读一",
      "passage": "包含编号空格的完整短文"
    }},
    {{
      "title": "阅读二",
      "passage": "包含编号空格的完整短文"
    }}
  ],
  "grammar": {{
    "title": "语法填空",
    "passage": "包含编号空格的完整语法填空短文"
  }},
  "answers": {{
    "readings": [
      {{
        "title": "阅读一答案",
        "items": [
          {{
            "number": "1",
            "answer": "答案",
            "explanation": "简短解析"
          }}
        ]
      }},
      {{
        "title": "阅读二答案",
        "items": []
      }}
    ],
    "grammar": [
      {{
        "number": "1",
        "answer": "答案",
        "kind": "有提示词",
        "prompt": "原词",
        "explanation": "简短解析"
      }}
    ]
  }}
}}

要求：
1. readings 必须正好包含两篇通顺、完整、符合高中生水平的高考真题风格短文；
2. 两篇阅读合计覆盖输入单元的全部目标词，每个目标词原则上只挖空一次；
3. 空格统一写成“(1) ______”“(2) ______”形式，两篇阅读连续编号；
4. 目标词可根据语法需要改变词形；
5. 阅读短文的空格处不提供首字母、中文释义、词性或其他提示；
6. grammar 必须是一篇符合江苏高考要求、且难度略高于普通高考真题的语法填空，设置10个空，单独从1开始编号；
7. grammar 必须同时覆盖三大从句考点：定语从句、名词性从句、状语从句；至少各设置1个空，并在 explanation 中标明具体考点；可以使用嵌套从句、长定语或插入成分提高难度，但不得影响语义判断；
8. grammar 必须设置“有提示词”和“无提示词”两类空，数量必须严格为：有提示词7题、无提示词3题；两类题目交错分布；
   - 有提示词7题：空格后必须保留括号提示词，例如“(1) ______ (use)”，答案重点考查名词、形容词、副词、时态、语态、非谓语和词形变化；至少包含2题较有区分度的词形转换，至少包含1题非谓语或被动结构；
   - 无提示词3题：空格后不得出现括号提示词，只从 a/an/the、定语从句连接词、名词性从句连接词、状语从句连接词、and、with、despite、all 等范围命题；三道无提示词不能集中在同一种考点；
9. 难度控制在“比普通高考略高但仍适合江苏高三课堂”：增加语篇逻辑、从句嵌套和结构干扰，不使用生僻词汇，不设置依赖专业背景的偏题；
10. grammar 正文中不得直接泄露答案；有提示词只能给出原词，不能给出答案形式；
11. answers.grammar 每题必须增加 kind 字段（“有提示词”或“无提示词”）和 prompt 字段（有提示词填写括号内原词，无提示词填写空字符串）；必须保证7个 kind 为“有提示词”、3个 kind 为“无提示词”；
12. 练习正文中不得泄露答案；所有答案只放在 answers 中；
13. explanation 每题只写一句简短解析，并说明词形变化、冠词、介词、连接词、非谓语或从句考点；
14. 只使用输入的本单元词汇作为目标词，不扩展无关内容；
15. readings、grammar、answers 所有字段必须存在；
16. 最终只输出可被 JSON.parse 直接解析的 JSON。"""
        return model_json_response(prompt, max_tokens=8000)
    except (ValueError, RuntimeError) as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    except Exception as exc:
        app.logger.exception("fill failed")
        return jsonify({"ok": False, "error": f"调用模型失败：{exc}"}), 502


@app.post("/api/reinforce")
def reinforce():
    try:
        data = read_json()
        word = require_text(data, "word")
        prompt = f"""【功能三：薄弱词/重点词强化】

目标单词：{word}

请只围绕这个单词生成适合江苏高三学生的强化内容，并严格返回一个合法 JSON 对象。
不要使用 Markdown 代码块，不要在 JSON 前后添加说明，不要输出注释。

固定结构如下，字段名必须完全一致：
{{
  "word": "目标单词",
  "phonetic": "音标",
  "partOfSpeech": "词性",
  "meaning": "核心中文词义",
  "examples": [
    {{
      "type": "高考语境例句",
      "english": "英文例句",
      "chinese": "中文翻译"
    }},
    {{
      "type": "真题风格例句",
      "english": "英文例句",
      "chinese": "中文翻译"
    }},
    {{
      "type": "幽默助记例句",
      "english": "英文例句",
      "chinese": "中文翻译"
    }}
  ],
  "collocations": [
    {{
      "phrase": "高频搭配",
      "meaning": "中文含义"
    }}
  ],
  "mnemonic": {{
    "method": "词根、词形、联想或情境助记",
    "tip": "简短准确的助记说明"
  }},
  "examTip": "一句话高考易错提醒"
}}

要求：
1. examples 必须正好包含高考语境、真题风格和健康幽默三个例句；幽默例句要有明显反差、夸张或机智包袱，不能只是普通日常句子；
2. 例句难度适合高中生，目标词用法准确，并提供自然的中文翻译；幽默例句要真正有笑点，但不得低俗、冒犯或依赖难懂双关；
3. 不得虚构具体高考年份、试卷名称或题目出处；
4. collocations 只列3—5个高中阶段最常用的搭配；
5. mnemonic 简短、准确、便于课堂记忆，不使用错误词源；
6. examTip 只写最值得提醒的一个易错点；
7. 不拓展无关同义词或无关知识；
8. 所有字段必须存在；
9. 最终只输出可被 JSON.parse 直接解析的 JSON。"""
        return model_json_response(prompt, max_tokens=3000)
    except (ValueError, RuntimeError) as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    except Exception as exc:
        app.logger.exception("reinforce failed")
        return jsonify({"ok": False, "error": f"调用模型失败：{exc}"}), 502


@app.post("/api/reinforce-batch")
def reinforce_batch():
    try:
        data = read_json()
        words = data.get("words")
        vocabulary = str(data.get("vocabulary", "")).strip()
        if not isinstance(words, list):
            raise ValueError("words 必须是单词数组。")

        clean_words: list[str] = []
        seen: set[str] = set()
        for item in words:
            word = str(item).strip()
            key = word.lower()
            if word and key not in seen:
                seen.add(key)
                clean_words.append(word)
        if not clean_words:
            raise ValueError("请至少提供一个单词。")
        if len(clean_words) > 8:
            raise ValueError("每批最多生成8个单词。")

        prompt = f"""【批量生成单词强化内容】

本批目标单词：{json.dumps(clean_words, ensure_ascii=False)}

原始单元词汇信息（用于参考音标、词性和释义）：
{vocabulary}

请为本批每个目标单词生成适合江苏高三学生的强化内容，并严格返回一个合法 JSON 对象。
不要使用 Markdown 代码块，不要在 JSON 前后添加说明，不要输出注释。

固定结构：
{{
  "items": [
    {{
      "word": "目标单词",
      "phonetic": "音标",
      "partOfSpeech": "词性",
      "meaning": "核心中文词义",
      "examples": [
        {{"type": "高考语境例句", "english": "英文例句", "chinese": "中文翻译"}},
        {{"type": "真题风格例句", "english": "英文例句", "chinese": "中文翻译"}},
        {{"type": "幽默助记例句", "english": "英文例句", "chinese": "中文翻译"}}
      ],
      "collocations": [
        {{"phrase": "高频搭配", "meaning": "中文含义"}}
      ],
      "mnemonic": {{"method": "助记方式", "tip": "简短助记说明"}},
      "examTip": "一句话高考易错提醒"
    }}
  ]
}}

要求：
1. items 数量必须与目标单词数量一致，并保持相同顺序；
2. 每个单词必须包含3个指定类型例句及中文翻译；其中幽默助记例句要有明显反差、夸张或机智包袱，不能只是普通日常句子；
3. 每个单词列出3—5个高中阶段高频搭配；
4. 不得虚构具体高考年份、试卷名称或题目出处；
5. 助记准确简洁，不使用错误词源；
6. 只处理本批目标单词；
7. 最终只输出可被 JSON.parse 直接解析的 JSON。"""
        return model_json_response(prompt, max_tokens=8000)
    except (ValueError, RuntimeError) as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    except Exception as exc:
        app.logger.exception("reinforce batch failed")
        return jsonify({"ok": False, "error": f"调用模型失败：{exc}"}), 502


@app.post("/api/weak-points-pdf")
def weak_points_pdf():
    """Generate a downloadable classroom weak-points PDF."""
    try:
        data = read_json()
        weak_words = data.get("weakWords")
        classroom_qa = data.get("classroomQA", [])
        title = str(data.get("title", "课堂薄弱点复习资料")).strip() or "课堂薄弱点复习资料"
        if not isinstance(weak_words, list) or not weak_words:
            raise ValueError("请先加入至少一个薄弱词。")
        if not isinstance(classroom_qa, list):
            classroom_qa = []
        if len(weak_words) > 100:
            raise ValueError("一次最多整理100个薄弱词。")

        word_items: list[str] = []
        for index, item in enumerate(weak_words, start=1):
            if not isinstance(item, dict):
                continue
            word = str(item.get("word", "")).strip()
            content = str(item.get("content", "")).strip()
            if word and content:
                word_items.append(f"{index}. {word}\n{content}")
        if not word_items:
            raise ValueError("薄弱点清单中没有可用的单词内容。")

        qa_items: list[tuple[str, str]] = []
        for index, item in enumerate(classroom_qa, start=1):
            if not isinstance(item, dict):
                continue
            question = str(item.get("question", "")).strip()
            answer = str(item.get("answer", "")).strip()
            if question and answer:
                qa_items.append((question, answer))

        article_prompt = f"""【生成课堂薄弱点复习文章】

本节课薄弱词及其强化信息：
{chr(10).join(word_items)}

请根据这些薄弱词及其高频短语，生成一篇针对性的高中英语高考真题风格挖词文章。
严格返回一个合法JSON对象，不要使用Markdown代码块，不要添加说明。

固定格式：
{{
  "title": "针对性词汇复习文章",
  "passage": "包含编号空格的英文文章",
  "answers": [
    {{"number": "1", "answer": "答案", "explanation": "简短解析"}}
  ]
}}

要求：
1. 尽可能自然地使用输入中的薄弱词及其短语；
2. 文章具有高考阅读或完形材料的语篇特点，主题积极健康；
3. 将重点薄弱词或短语挖空，空格统一使用“(1) ______”形式；
4. 文章正文不得直接泄露答案或中文解释；
5. answers 只保留文章中实际设置的空格，编号与正文一致；
6. 文章长度控制在350—500词；
7. 只输出可被JSON.parse直接解析的JSON。"""
        article_raw = collect_json_model(article_prompt, max_tokens=3000)
        cleaned = article_raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.replace("```json", "", 1).replace("```", "", 1).strip()
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start < 0 or end <= start:
            raise RuntimeError("模型返回的文章格式不正确。")
        article_data = json.loads(cleaned[start:end + 1])
        article_title = str(article_data.get("title", "针对性词汇复习文章"))
        passage = str(article_data.get("passage", "")).strip()
        article_answers = article_data.get("answers", [])
        if not isinstance(article_answers, list):
            article_answers = []
        if not passage:
            raise RuntimeError("模型没有返回复习文章。")

        pdf_bytes = build_weak_points_pdf(title, word_items, article_title, passage, article_answers, qa_items)
        filename = f"课堂薄弱点复习资料_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf"
        return send_file(
            io.BytesIO(pdf_bytes),
            mimetype="application/pdf",
            as_attachment=True,
            download_name=filename,
        )
    except (ValueError, RuntimeError, json.JSONDecodeError) as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    except Exception as exc:
        app.logger.exception("weak points pdf failed")
        return jsonify({"ok": False, "error": f"生成PDF失败：{exc}"}), 502


def build_weak_points_pdf(
    title: str,
    word_items: list[str],
    article_title: str,
    passage: str,
    article_answers: list[Any],
    qa_items: list[tuple[str, str]],
) -> bytes:
    try:
        from reportlab.lib import colors
        from reportlab.lib.enums import TA_CENTER
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import mm
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
        from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
    except ImportError as exc:
        raise RuntimeError("缺少PDF生成依赖，请先安装reportlab。") from exc

    chinese_font_candidates = [
        r"C:\Windows\Fonts\simsun.ttc",
        r"C:\Windows\Fonts\msyh.ttc",
    ]
    times_font = r"C:\Windows\Fonts\times.ttf"
    chinese_path = next((path for path in chinese_font_candidates if Path(path).exists()), None)
    if not chinese_path or not Path(times_font).exists():
        raise RuntimeError("系统中未找到可用的Times New Roman或中文字体。")
    try:
        pdfmetrics.registerFont(TTFont("ClassroomChinese", chinese_path, subfontIndex=0))
    except TypeError:
        pdfmetrics.registerFont(TTFont("ClassroomChinese", chinese_path))
    pdfmetrics.registerFont(TTFont("TimesNewRoman", times_font))

    def clean_pdf_text(text: Any) -> str:
        value = str(text or "")
        value = re.sub(r"```(?:markdown|text|json)?", "", value, flags=re.IGNORECASE)
        value = value.replace("```", "").replace("**", "").replace("__", "").replace("`", "")
        value = re.sub(r"\[[^\]]+\]", "", value)
        value = re.sub(r"^\s*#{1,6}\s*", "", value, flags=re.MULTILINE)
        value = value.replace("\r\n", "\n").replace("\r", "\n")
        return value.strip()

    def safe(text: Any) -> str:
        value = clean_pdf_text(text)
        return value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br/>")

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=16 * mm,
        leftMargin=16 * mm,
        topMargin=15 * mm,
        bottomMargin=15 * mm,
        title=title,
        author="made by 昆山市高级中学 申珅",
    )
    styles = getSampleStyleSheet()
    navy = colors.HexColor("#172554")
    blue = colors.HexColor("#3157d5")
    light_blue = colors.HexColor("#eff6ff")
    border_blue = colors.HexColor("#bfdbfe")
    body_color = colors.HexColor("#334155")
    title_style = ParagraphStyle("PdfTitle", parent=styles["Title"], fontName="ClassroomChinese", fontSize=19, leading=23, alignment=TA_CENTER, textColor=navy, spaceAfter=4)
    sub_style = ParagraphStyle("PdfSub", parent=styles["Normal"], fontName="ClassroomChinese", fontSize=8.5, leading=11, alignment=TA_CENTER, textColor=colors.HexColor("#64748b"), spaceAfter=9)
    section_style = ParagraphStyle("PdfSection", parent=styles["Heading2"], fontName="ClassroomChinese", fontSize=13, leading=17, textColor=blue, spaceBefore=0, spaceAfter=6)
    card_title_style = ParagraphStyle("PdfCardTitle", parent=styles["BodyText"], fontName="ClassroomChinese", fontSize=9.5, leading=13, textColor=navy)
    body_style = ParagraphStyle("PdfBody", parent=styles["BodyText"], fontName="ClassroomChinese", fontSize=9, leading=13, textColor=body_color, spaceAfter=3)
    small_style = ParagraphStyle("PdfSmall", parent=body_style, fontSize=8.2, leading=11, textColor=colors.HexColor("#475569"))
    article_style = ParagraphStyle("PdfArticle", parent=styles["BodyText"], fontName="TimesNewRoman", fontSize=10, leading=15, textColor=body_color, spaceAfter=3)
    answer_style = ParagraphStyle("PdfAnswer", parent=styles["BodyText"], fontName="TimesNewRoman", fontSize=8.5, leading=11, textColor=colors.HexColor("#1f7a3a"))

    def card(rows: list[list[Any]], header_color=light_blue, body_color_hex="#fbfdff") -> Table:
        table = Table(rows, colWidths=[178 * mm])
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), header_color),
            ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor(body_color_hex)),
            ("BOX", (0, 0), (-1, -1), 0.6, border_blue),
            ("LINEBELOW", (0, 0), (-1, 0), 0.4, colors.HexColor("#dbeafe")),
            ("LEFTPADDING", (0, 0), (-1, -1), 9), ("RIGHTPADDING", (0, 0), (-1, -1), 9),
            ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        return table

    story = [Paragraph(safe(title), title_style), Paragraph("江苏高三英语课堂复习资料 · made by 昆山市高级中学 申珅", sub_style)]
    story.append(Paragraph("一、薄弱词强化", section_style))
    for item in word_items:
        parts = item.split("\n", 1)
        word_title = parts[0]
        content = parts[1] if len(parts) > 1 else ""
        story.extend([
            card([[Paragraph(f"<b>{safe(word_title)}</b>", card_title_style)], [Paragraph(safe(content), small_style)]]),
            Spacer(1, 2),
        ])

    story.append(PageBreak())
    story.append(Paragraph("二、针对性高考真题风格文章", section_style))
    story.append(card([
        [Paragraph(safe(article_title), card_title_style)],
        [Paragraph(safe(passage), article_style)],
    ]))
    story.append(Spacer(1, 5))
    story.append(Paragraph("参考答案", ParagraphStyle("AnswerHeading", parent=section_style, fontName="ClassroomChinese", fontSize=11, leading=15, textColor=blue, spaceBefore=0)))
    answer_rows = [[Paragraph("题号", small_style), Paragraph("答案", small_style), Paragraph("简析", small_style)]]
    for item in article_answers:
        if not isinstance(item, dict):
            continue
        answer_rows.append([
            Paragraph(safe(item.get("number", "")), small_style),
            Paragraph(safe(item.get("answer", "")), answer_style),
            Paragraph(safe(item.get("explanation", "")), small_style),
        ])
    if len(answer_rows) == 1:
        answer_rows.append([Paragraph("-", small_style), Paragraph("暂无答案", small_style), Paragraph("-", small_style)])
    answer_table = Table(answer_rows, colWidths=[18 * mm, 42 * mm, 118 * mm], repeatRows=1)
    answer_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), light_blue),
        ("GRID", (0, 0), (-1, -1), 0.4, border_blue),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 7), ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(answer_table)

    story.append(PageBreak())
    story.append(Paragraph("三、课堂提问与回答", section_style))
    if qa_items:
        for index, (question, answer) in enumerate(qa_items, start=1):
            story.extend([
                card([
                    [Paragraph(f"<b>提问 {index}</b>", card_title_style)],
                    [Paragraph(f"<b>问题：</b>{safe(question)}<br/><b>回答：</b>{safe(answer)}", small_style)],
                ]),
                Spacer(1, 3),
            ])
    else:
        story.append(Paragraph("本节课未记录课堂提问。", body_style))

    def draw_footer(canvas, document):
        canvas.saveState()
        canvas.setFont("ClassroomChinese", 8)
        canvas.setFillColor(colors.HexColor("#94a3b8"))
        canvas.drawString(16 * mm, 8 * mm, "词小灶 · made by 昆山市高级中学 申珅")
        canvas.drawRightString(A4[0] - 16 * mm, 8 * mm, f"第 {document.page} 页")
        canvas.restoreState()

    doc.build(story, onFirstPage=draw_footer, onLaterPages=draw_footer)
    return buffer.getvalue()


@app.post("/api/summary")
def summary():
    try:
        data = read_json()
        weak_words = data.get("weakWords")
        if not isinstance(weak_words, list) or not weak_words:
            raise ValueError("请先加入至少一个薄弱词或重点词。")
        if len(weak_words) > 100:
            raise ValueError("一次最多整理100个词。")

        items: list[str] = []
        for index, item in enumerate(weak_words, start=1):
            if not isinstance(item, dict):
                continue
            word = str(item.get("word", "")).strip()
            content = str(item.get("content", "")).strip()
            if word and content:
                items.append(f"### {index}. {word}\n{content}")
        if not items:
            raise ValueError("薄弱词清单中没有可用内容。")

        prompt = f"""【生成本单元薄弱词/重点词复习资料】

以下是教师在课堂中记录的薄弱词信息：
{chr(10).join(items)}

请整理成一份可以直接发给学生的复习资料：
1. 按单词分别整理词性、核心词义、高频搭配、例句和助记技巧；
2. 删除重复内容，保留最有用的信息；
3. 最后增加一段简短的本单元复习总结和复习建议；
4. 不增加输入清单之外的无关词汇；
5. 使用清晰的Markdown标题，语言简洁。"""
        return model_stream_response(prompt)
    except (ValueError, RuntimeError) as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    except Exception as exc:
        app.logger.exception("summary failed")
        return jsonify({"ok": False, "error": f"调用模型失败：{exc}"}), 502


if __name__ == "__main__":
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "5000"))
    if getattr(sys, "frozen", False):
        # 打包版双击后自动打开浏览器；开发环境保持原有启动方式。
        threading.Timer(1.2, lambda: webbrowser.open(f"http://{host}:{port}")).start()
    app.run(host=host, port=port, debug=False)
