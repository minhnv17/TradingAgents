

from tradingagents.agents.utils.agent_utils import get_language_instruction

def create_neutral_debator(llm):
    def neutral_node(state) -> dict:
        risk_debate_state = state["risk_debate_state"]
        history = risk_debate_state.get("history", "")
        neutral_history = risk_debate_state.get("neutral_history", "")

        current_aggressive_response = risk_debate_state.get("current_aggressive_response", "")
        current_conservative_response = risk_debate_state.get("current_conservative_response", "")

        market_research_report = state["market_report"]
        sentiment_report = state["sentiment_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]

        trader_decision = state["trader_investment_plan"]

        prompt = f"""Bạn là **Neutral Risk Analyst**: nhiệm vụ là đưa góc nhìn cân bằng, cân đối lợi ích và rủi ro của đề xuất Trader. Bạn đánh giá cả upside lẫn downside, cân nhắc xu hướng thị trường rộng hơn, thay đổi vĩ mô có thể xảy ra và chiến lược đa dạng hoá.{get_language_instruction()}

Đây là đề xuất của Trader:

{trader_decision}

Nhiệm vụ của bạn là thách thức cả Aggressive lẫn Conservative: chỉ ra chỗ mỗi bên quá lạc quan hoặc quá thận trọng. Dựa trên các nguồn sau để đề xuất một chiến lược “vừa phải” và bền vững hơn (nếu cần) nhằm điều chỉnh đề xuất Trader:

Market Research Report: {market_research_report}
Social Media Sentiment Report: {sentiment_report}
Latest World Affairs Report: {news_report}
Company Fundamentals Report: {fundamentals_report}
Lịch sử tranh luận hiện tại: {history}
Phản hồi gần nhất của Aggressive: {current_aggressive_response}
Phản hồi gần nhất của Conservative: {current_conservative_response}
Nếu chưa có phản hồi từ bên khác, hãy tự trình bày luận điểm của bạn dựa trên dữ liệu sẵn có.

Yêu cầu: tranh luận, phản biện điểm yếu của cả hai bên; làm rõ vì sao chiến lược rủi ro vừa phải có thể “được cả hai” (có tăng trưởng nhưng tránh biến động cực đoan). Tập trung vào lập luận chứ không chỉ liệt kê dữ liệu. Trình bày dạng hội thoại tự nhiên, không cần định dạng đặc biệt."""

        response = llm.invoke(prompt)

        argument = f"Neutral Analyst: {response.content}"

        new_risk_debate_state = {
            "history": history + "\n" + argument,
            "aggressive_history": risk_debate_state.get("aggressive_history", ""),
            "conservative_history": risk_debate_state.get("conservative_history", ""),
            "neutral_history": neutral_history + "\n" + argument,
            "latest_speaker": "Neutral",
            "current_aggressive_response": risk_debate_state.get(
                "current_aggressive_response", ""
            ),
            "current_conservative_response": risk_debate_state.get("current_conservative_response", ""),
            "current_neutral_response": argument,
            "count": risk_debate_state["count"] + 1,
        }

        return {"risk_debate_state": new_risk_debate_state}

    return neutral_node
