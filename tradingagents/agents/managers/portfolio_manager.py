"""Portfolio Manager: synthesises the risk-analyst debate into the final decision.

Uses LangChain's ``with_structured_output`` so the LLM produces a typed
``PortfolioDecision`` directly, in a single call.  The result is rendered
back to markdown for storage in ``final_trade_decision`` so memory log,
CLI display, and saved reports continue to consume the same shape they do
today.  When a provider does not expose structured output, the agent falls
back gracefully to free-text generation.
"""

from __future__ import annotations

from tradingagents.agents.schemas import PortfolioDecision, render_pm_decision
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


def create_portfolio_manager(llm):
    structured_llm = bind_structured(llm, PortfolioDecision, "Portfolio Manager")

    def portfolio_manager_node(state) -> dict:
        instrument_context = build_instrument_context(state["company_of_interest"])

        history = state["risk_debate_state"]["history"]
        risk_debate_state = state["risk_debate_state"]
        research_plan = state["investment_plan"]
        trader_plan = state["trader_investment_plan"]

        config = get_config()
        language = config.get("output_language", "English")
        lang_code = get_language_code(language)

        past_context = state.get("past_context", "")
        lessons_line = {
            "en": (
                f"- Lessons from prior decisions and outcomes:\n{past_context}\n"
                if past_context
                else ""
            ),
            "vi": (
                f"- Bài học từ các quyết định và kết quả trước đó:\n{past_context}\n"
                if past_context
                else ""
            ),
        }

        rating_scale = {
            "en": (
                "**Rating Scale** (use exactly one):\n"
                "- **Buy**: Strong conviction to enter or add to position\n"
                "- **Overweight**: Favorable outlook, gradually increase exposure\n"
                "- **Hold**: Maintain current position, no action needed\n"
                "- **Underweight**: Reduce exposure, take partial profits\n"
                "- **Sell**: Exit position or avoid entry"
            ),
            "vi": (
                "**Thang xếp hạng** (chọn một):\n"
                "- **Mua**: Tin tưởng mạnh để mở vị thế hoặc thêm vị thế\n"
                "- **Tăng Tỷ Trọng**: Triển vọng tích cực, tăng dần tỷ trọng\n"
                "- **Nắm Giữ**: Giữ nguyên vị thế, không cần hành động\n"
                "- **Giảm Tỷ Trọng**: Giảm tỷ trọng, chốt lời một phần\n"
                "- **Bán**: Đóng vị thế hoặc tránh mở vị thế"
            ),
        }

        context_labels = {
            "en": {
                "research_plan": "Research Manager's investment plan",
                "trader_plan": "Trader's transaction proposal",
                "debate_history": "Risk Analysts Debate History",
            },
            "vi": {
                "research_plan": "Kế hoạch đầu tư của Research Manager",
                "trader_plan": "Đề xuất giao dịch của Trader",
                "debate_history": "Lịch sử tranh luận các chuyên gia phân tích rủi ro",
            },
        }
        lbl = context_labels.get(lang_code, context_labels["en"])

        prompt = f"""Bạn là Portfolio Manager. Hãy tổng hợp cuộc tranh luận của nhóm Risk Analyst và đưa ra quyết định giao dịch cuối cùng.

{instrument_context}

---

{rating_scale.get(lang_code, rating_scale["en"])}

**Context:**
- {lbl["research_plan"]}: **{research_plan}**
- {lbl["trader_plan"]}: **{trader_plan}**
{lessons_line.get(lang_code, lessons_line["en"])}
**{lbl["debate_history"]}:**
{history}

---

Hãy dứt khoát và dựa mọi kết luận vào bằng chứng cụ thể từ các báo cáo/phân tích đã có.{get_language_instruction()}"""

        final_trade_decision = invoke_structured_or_freetext(
            structured_llm,
            llm,
            prompt,
            render_pm_decision,
            "Portfolio Manager",
        )

        new_risk_debate_state = {
            "judge_decision": final_trade_decision,
            "history": risk_debate_state["history"],
            "aggressive_history": risk_debate_state["aggressive_history"],
            "conservative_history": risk_debate_state["conservative_history"],
            "neutral_history": risk_debate_state["neutral_history"],
            "latest_speaker": "Judge",
            "current_aggressive_response": risk_debate_state[
                "current_aggressive_response"
            ],
            "current_conservative_response": risk_debate_state[
                "current_conservative_response"
            ],
            "current_neutral_response": risk_debate_state["current_neutral_response"],
            "count": risk_debate_state["count"],
        }

        return {
            "risk_debate_state": new_risk_debate_state,
            "final_trade_decision": final_trade_decision,
        }

    return portfolio_manager_node
