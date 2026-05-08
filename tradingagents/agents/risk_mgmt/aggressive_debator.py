

from tradingagents.agents.utils.agent_utils import get_language_instruction

def create_aggressive_debator(llm):
    def aggressive_node(state) -> dict:
        risk_debate_state = state["risk_debate_state"]
        history = risk_debate_state.get("history", "")
        aggressive_history = risk_debate_state.get("aggressive_history", "")

        current_conservative_response = risk_debate_state.get("current_conservative_response", "")
        current_neutral_response = risk_debate_state.get("current_neutral_response", "")

        market_research_report = state["market_report"]
        sentiment_report = state["sentiment_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]

        trader_decision = state["trader_investment_plan"]

        prompt = f"""Bạn là **Aggressive Risk Analyst**: ưu tiên cơ hội lợi nhuận cao, chấp nhận rủi ro cao khi phần thưởng xứng đáng. Khi đánh giá kế hoạch/đề xuất của Trader, hãy tập trung vào upside, động lực tăng trưởng, và các lợi thế/catalyst—even nếu rủi ro cao. Dùng dữ liệu thị trường và sentiment để củng cố luận điểm và phản biện các quan điểm đối lập. Hãy trả lời trực tiếp từng ý của Conservative và Neutral, dùng phản biện có số liệu/logic thuyết phục và chỉ ra chỗ họ quá thận trọng, bỏ lỡ cơ hội.{get_language_instruction()}

Đây là đề xuất của Trader:

{trader_decision}

Nhiệm vụ của bạn là bảo vệ (hoặc điều chỉnh theo hướng “mạnh tay hơn”) đề xuất của Trader bằng cách chất vấn/đánh vào điểm yếu của quan điểm Conservative và Neutral. Hãy kết hợp các nguồn sau:

Market Research Report: {market_research_report}
Social Media Sentiment Report: {sentiment_report}
Latest World Affairs Report: {news_report}
Company Fundamentals Report: {fundamentals_report}
Lịch sử tranh luận hiện tại: {history}
Lập luận gần nhất của Conservative: {current_conservative_response}
Lập luận gần nhất của Neutral: {current_neutral_response}
Nếu chưa có phản hồi từ bên khác, hãy tự trình bày luận điểm của bạn dựa trên dữ liệu sẵn có.

Yêu cầu: tranh luận và thuyết phục (không chỉ liệt kê số liệu). Hãy phản biện từng counterpoint để làm rõ vì sao cách tiếp cận “risk-on” là tối ưu trong bối cảnh này. Trình bày dạng hội thoại tự nhiên, không cần định dạng đặc biệt."""

        response = llm.invoke(prompt)

        argument = f"Aggressive Analyst: {response.content}"

        new_risk_debate_state = {
            "history": history + "\n" + argument,
            "aggressive_history": aggressive_history + "\n" + argument,
            "conservative_history": risk_debate_state.get("conservative_history", ""),
            "neutral_history": risk_debate_state.get("neutral_history", ""),
            "latest_speaker": "Aggressive",
            "current_aggressive_response": argument,
            "current_conservative_response": risk_debate_state.get("current_conservative_response", ""),
            "current_neutral_response": risk_debate_state.get(
                "current_neutral_response", ""
            ),
            "count": risk_debate_state["count"] + 1,
        }

        return {"risk_debate_state": new_risk_debate_state}

    return aggressive_node
