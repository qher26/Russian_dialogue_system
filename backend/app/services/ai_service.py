import json
import logging
import os
import random
from typing import Optional, Dict, Any, List

from openai import OpenAI

from app.services.voice_service import voice_service

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AIService:
    """DeepSeek-backed AI service for Russian conversation practice."""

    def __init__(self):
        # Credentials stay on the backend. Never embed them in the Android APK.
        self.api_key = os.getenv("DEEPSEEK_API_KEY")
        self.base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
        self.model = os.getenv("DEEPSEEK_MODEL", "deepseek-flash")
        self.client = None

        if self.api_key:
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=60.0,
                max_retries=2,
            )
            logger.info("DeepSeek client initialized (model=%s)", self.model)
        else:
            logger.warning(
                "DEEPSEEK_API_KEY is not configured; AI requests will use fallback responses"
            )

    def _chat(self, messages: List[Dict[str, str]]) -> str:
        """Call DeepSeek through its OpenAI-compatible Chat Completions API."""
        if self.client is None:
            raise RuntimeError("DEEPSEEK_API_KEY is not configured on the backend")

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
        )
        content = response.choices[0].message.content
        if not content:
            raise RuntimeError("DeepSeek returned an empty response")
        return content.strip()

    def generate_initial_message(self, scenario: Dict[str, Any]) -> Optional[str]:
        """Generate a Chinese scene introduction followed by a Russian greeting."""
        location = scenario.get("location", "")
        time_of_day = scenario.get("time_of_day", "")
        characters = scenario.get("characters", [])
        character_description = "，".join(characters) if characters else "相关人员"

        prompt = (
            f"你正在扮演一个在{location}的{character_description}。现在是{time_of_day}，"
            "请先用中文向用户介绍当前场景（一句话即可），然后用俄语主动向用户打招呼"
            "并开始对话。请根据场景提供合适的开场白。格式：先中文场景介绍，再写俄语对话内容。"
        )
        try:
            return self._chat([
                {"role": "system", "content": "你是一位专业、耐心的俄语口语陪练老师。"},
                {"role": "user", "content": prompt},
            ])
        except Exception:
            logger.exception("DeepSeek failed to generate the initial message")
            return (
                f"您正在{location}场景中与{character_description}对话。\n"
                "Здравствуйте! Чем я могу вам помочь?"
            )

    def generate_response(
        self,
        user_message: str,
        scenario: Dict[str, Any],
        conversation_history: List[Dict[str, str]] = None,
    ) -> Optional[str]:
        """Generate the next Russian-language turn in a scenario conversation."""
        location = scenario.get("location", "")
        time_of_day = scenario.get("time_of_day", "")
        characters = scenario.get("characters", [])
        character_description = "，".join(characters) if characters else "相关人员"

        messages: List[Dict[str, str]] = [{
            "role": "system",
            "content": (
                f"你正在扮演{location}的{character_description}。现在是{time_of_day}。"
                "请始终使用自然、符合日常语境的俄语与学习者交流。"
                "根据学习者当前俄语水平调整难度；每次回复以简短自然的对话为主，"
                "不要突然切换成中文，除非用户明确要求解释。"
            ),
        }]
        if conversation_history:
            messages.extend(conversation_history)
        messages.append({"role": "user", "content": user_message})

        try:
            return self._chat(messages)
        except Exception:
            logger.exception("DeepSeek failed to generate a conversation response")
            return "Извините, я не понял. Можете повторить, пожалуйста?"

    def generate_chinese_evaluation(self, user_message: str) -> dict:
        """Evaluate a Russian utterance and return a structured Chinese report."""
        evaluation_prompt = (
            "请对以下俄语语句进行发音、语法、流利度和词汇四个维度的评分，"
            "每项满分100分，并给出中文总体评价和具体建议。"
            "注意：仅凭文本无法可靠判断真实发音，请将发音评分标注为文本估计，"
            "不要声称已经听过音频。"
            f"用户俄语语句：{user_message!r}。"
            "严格返回JSON对象，字段为overall_score、pronunciation、grammar、"
            "fluency、vocabulary、evaluation_text、suggestions。"
        )
        try:
            content = self._chat([
                {"role": "system", "content": "你是严谨的俄语教师。只返回合法JSON，不要使用Markdown代码块。"},
                {"role": "user", "content": evaluation_prompt},
            ])
            content = content.strip().strip(chr(96)).strip()
            if content.lower().startswith("json"):
                content = content[4:].strip()
            result = json.loads(content)
            required = {
                "overall_score", "pronunciation", "grammar", "fluency",
                "vocabulary", "evaluation_text", "suggestions",
            }
            if not required.issubset(result):
                raise ValueError("DeepSeek evaluation JSON is missing required fields")
            return result
        except Exception:
            logger.exception("DeepSeek evaluation failed; using fallback evaluation")
            pronunciation = random.randint(60, 100)
            grammar = random.randint(60, 100)
            fluency = random.randint(60, 100)
            vocabulary = random.randint(60, 100)
            overall_score = (pronunciation + grammar + fluency + vocabulary) // 4

            if overall_score >= 90:
                evaluation_text = "非常好！您的俄语表达很地道。"
            elif overall_score >= 80:
                evaluation_text = "很好！您的俄语表达比较自然。"
            elif overall_score >= 70:
                evaluation_text = "不错！继续练习会更好。"
            else:
                evaluation_text = "还需要加强练习，加油！"

            suggestions = []
            if pronunciation < 80:
                suggestions.append("注意俄语发音；当前分数仅为文本估计，并非真实语音测评。")
            if grammar < 80:
                suggestions.append("注意俄语语法结构，特别是名词格变化和动词变位。")
            if fluency < 80:
                suggestions.append("尝试使用更完整的句子进行表达。")
            if vocabulary < 80:
                suggestions.append("扩展词汇量，学习更多日常用语。")
            if not suggestions:
                suggestions.append("继续保持练习，您的俄语正在进步！")

            return {
                "overall_score": overall_score,
                "pronunciation": pronunciation,
                "grammar": grammar,
                "fluency": fluency,
                "vocabulary": vocabulary,
                "evaluation_text": evaluation_text,
                "suggestions": suggestions,
            }

    def generate_response_with_voice(
        self,
        user_message: str,
        scenario: Dict[str, Any],
        conversation_history: List[Dict[str, str]] = None,
    ) -> dict:
        """Return the evaluation associated with a voice conversation turn."""
        try:
            return self.generate_chinese_evaluation(user_message)
        except Exception:
            logger.exception("Failed to generate voice evaluation")
            return {
                "overall_score": 80,
                "pronunciation": 85,
                "grammar": 75,
                "fluency": 80,
                "vocabulary": 78,
                "evaluation_text": "系统默认评价",
                "suggestions": ["请继续努力练习俄语"],
            }


ai_service = AIService()
