"""Research Manager: turns the bull/bear debate into a structured investment plan for the trader."""

from __future__ import annotations

from tradingagents.agents.schemas import ResearchPlan, render_research_plan
from tradingagents.agents.utils.agent_utils import (
    build_instrument_context,
    get_language_instruction,
)
from tradingagents.agents.utils.structured import (
    bind_structured,
    invoke_structured_or_freetext,
)
from tradingagents.dataflows.config import get_config
from tradingagents.localization import get_language_code


def create_research_manager(llm):
    structured_llm = bind_structured(llm, ResearchPlan, "Research Manager")

    def research_manager_node(state) -> dict:
        instrument_context = build_instrument_context(state["company_of_interest"])
        history = state["investment_debate_state"].get("history", "")

        investment_debate_state = state["investment_debate_state"]

        config = get_config()
        language = config.get("output_language", "English")
        lang_code = get_language_code(language)

        rating_scale = {
            "en": (
                "**Rating Scale** (use exactly one):\n"
                "- **Buy**: Strong conviction in the bull thesis; recommend taking or growing the position\n"
                "- **Overweight**: Constructive view; recommend gradually increasing exposure\n"
                "- **Hold**: Balanced view; recommend maintaining the current position\n"
                "- **Underweight**: Cautious view; recommend trimming exposure\n"
                "- **Sell**: Strong conviction in the bear thesis; recommend exiting or avoiding the position\n\n"
                "Commit to a clear stance whenever the debate's strongest arguments warrant one; reserve Hold for situations where the evidence on both sides is genuinely balanced."
            ),
            "vi": (
                "**Thang xếp hạng** (chọn một):\n"
                "- **Mua**: Tin tưởng mạnh vào luận điểm tăng giá; khuyến nghị mở vị thế hoặc tăng vị thế\n"
                "- **Tăng Tỷ Trọng**: Góc nhìn tích cực; khuyến nghị tăng dần tỷ trọng\n"
                "- **Nắm Giữ**: Góc nhìn cân bằng; khuyến nghị giữ nguyên vị thế\n"
                "- **Giảm Tỷ Trọng**: Góc nhìn thận trọng; khuyến nghị giảm tỷ trọng\n"
                "- **Bán**: Tin tưởng mạnh vào luận điểm giảm giá; khuyến nghị đóng vị thế hoặc tránh\n\n"
                "Hãy đưa ra quan điểm rõ ràng khi các luận điểm mạnh nhất của cuộc tranh luận xứng đáng điều đó; chỉ dùng Nắm Giữ khi bằng chứng từ hai phía thực sự cân bằng."
            ),
        }

        prompt = f"""Bạn là Research Manager và người điều phối tranh luận. Nhiệm vụ là đánh giá phản biện (một cách nghiêm khắc nhưng công bằng) các luận điểm Bull vs Bear, sau đó đưa ra một kế hoạch đầu tư rõ ràng, có thể hành động được cho Trader.{get_language_instruction()}

{instrument_context}

---

{rating_scale.get(lang_code, rating_scale["en"])}

---

**Debate History:**
{history}"""

        investment_plan = invoke_structured_or_freetext(
            structured_llm,
            llm,
            prompt,
            render_research_plan,
            "Research Manager",
        )

        new_investment_debate_state = {
            "judge_decision": investment_plan,
            "history": investment_debate_state.get("history", ""),
            "bear_history": investment_debate_state.get("bear_history", ""),
            "bull_history": investment_debate_state.get("bull_history", ""),
            "current_response": investment_plan,
            "count": investment_debate_state["count"],
        }

        return {
            "investment_debate_state": new_investment_debate_state,
            "investment_plan": investment_plan,
        }

    return research_manager_node
