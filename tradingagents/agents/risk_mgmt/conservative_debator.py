

from tradingagents.agents.utils.agent_utils import get_language_instruction

def create_conservative_debator(llm):
    def conservative_node(state) -> dict:
        risk_debate_state = state["risk_debate_state"]
        history = risk_debate_state.get("history", "")
        conservative_history = risk_debate_state.get("conservative_history", "")

        current_aggressive_response = risk_debate_state.get("current_aggressive_response", "")
        current_neutral_response = risk_debate_state.get("current_neutral_response", "")

        market_research_report = state["market_report"]
        sentiment_report = state["sentiment_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]

        trader_decision = state["trader_investment_plan"]

        prompt = f"""Bạn là **Conservative Risk Analyst**: ưu tiên bảo toàn vốn, giảm biến động, và tăng trưởng ổn định. Khi đánh giá kế hoạch/đề xuất của Trader, hãy soi kỹ các yếu tố rủi ro cao: đâu là điểm có thể gây drawdown lớn, rủi ro vĩ mô/chính sách, rủi ro dòng tiền/đòn bẩy, và nơi mà phương án thận trọng hơn sẽ tốt cho lợi nhuận điều chỉnh theo rủi ro.{get_language_instruction()}

Đây là đề xuất của Trader:

{trader_decision}

Nhiệm vụ của bạn là phản biện Aggressive và Neutral: chỉ ra chỗ họ bỏ qua rủi ro, hoặc đánh đổi an toàn quá nhiều để lấy upside. Trả lời trực tiếp từng ý, dựa trên các nguồn sau để đề xuất điều chỉnh theo hướng rủi ro thấp:

Market Research Report: {market_research_report}
Social Media Sentiment Report: {sentiment_report}
Latest World Affairs Report: {news_report}
Company Fundamentals Report: {fundamentals_report}
Lịch sử tranh luận hiện tại: {history}
Phản hồi gần nhất của Aggressive: {current_aggressive_response}
Phản hồi gần nhất của Neutral: {current_neutral_response}
Nếu chưa có phản hồi từ bên khác, hãy tự trình bày luận điểm của bạn dựa trên dữ liệu sẵn có.

Yêu cầu: tranh luận và chất vấn trực tiếp; nhấn mạnh downside mà họ bỏ qua; kết luận rõ vì sao cách tiếp cận bảo thủ là con đường an toàn nhất cho tài sản của “firm”. Trình bày dạng hội thoại tự nhiên, không cần định dạng đặc biệt."""

        response = llm.invoke(prompt)

        argument = f"Conservative Analyst: {response.content}"

        new_risk_debate_state = {
            "history": history + "\n" + argument,
            "aggressive_history": risk_debate_state.get("aggressive_history", ""),
            "conservative_history": conservative_history + "\n" + argument,
            "neutral_history": risk_debate_state.get("neutral_history", ""),
            "latest_speaker": "Conservative",
            "current_aggressive_response": risk_debate_state.get(
                "current_aggressive_response", ""
            ),
            "current_conservative_response": argument,
            "current_neutral_response": risk_debate_state.get(
                "current_neutral_response", ""
            ),
            "count": risk_debate_state["count"] + 1,
        }

        return {"risk_debate_state": new_risk_debate_state}

    return conservative_node
