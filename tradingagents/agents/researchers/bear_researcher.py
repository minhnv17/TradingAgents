from tradingagents.agents.utils.agent_utils import get_language_instruction

def create_bear_researcher(llm):
    def bear_node(state) -> dict:
        investment_debate_state = state["investment_debate_state"]
        history = investment_debate_state.get("history", "")
        bear_history = investment_debate_state.get("bear_history", "")

        current_response = investment_debate_state.get("current_response", "")
        market_research_report = state["market_report"]
        sentiment_report = state["sentiment_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]

        prompt = f"""Bạn là nhà phân tích theo hướng “Bear” (phản đối đầu tư vào cổ phiếu). Mục tiêu là trình bày một luận điểm chặt chẽ, nhấn mạnh rủi ro, thách thức và các tín hiệu tiêu cực có thể khiến hiệu suất cổ phiếu kém.{get_language_instruction()}

Yêu cầu nội dung:
- **Rủi ro & thách thức**: bão hoà thị trường, chu kỳ ngành, cấu trúc nợ, dòng tiền, rủi ro pháp lý/chính sách, rủi ro vĩ mô, v.v.
- **Điểm yếu cạnh tranh**: vị thế yếu hơn đối thủ, mất lợi thế chi phí, suy giảm đổi mới, rào cản gia nhập thấp, v.v.
- **Tín hiệu tiêu cực**: dữ liệu tài chính xấu đi, xu hướng thị trường bất lợi, tin tức tiêu cực, rủi ro sự kiện.
- **Phản biện Bull**: bắt lỗi các giả định quá lạc quan của Bull bằng dữ liệu/chi tiết cụ thể; chỉ ra “điểm mù” trong luận cứ.
- **Cách trình bày**: tranh luận đối thoại trực tiếp, phản hồi các ý của Bull (không chỉ liệt kê).

Tài nguyên có sẵn:
- Market research report: {market_research_report}
- Social media sentiment report: {sentiment_report}
- Latest world affairs news: {news_report}
- Company fundamentals report: {fundamentals_report}
- Lịch sử tranh luận: {history}
- Luận điểm Bull gần nhất: {current_response}

Hãy dùng các tài nguyên trên để đưa ra lập luận Bear, phản biện Bull, và làm rõ vì sao rủi ro/nhược điểm đang áp đảo cơ hội.
"""

        response = llm.invoke(prompt)

        argument = f"Bear Analyst: {response.content}"

        new_investment_debate_state = {
            "history": history + "\n" + argument,
            "bear_history": bear_history + "\n" + argument,
            "bull_history": investment_debate_state.get("bull_history", ""),
            "current_response": argument,
            "count": investment_debate_state["count"] + 1,
        }

        return {"investment_debate_state": new_investment_debate_state}

    return bear_node
