from tradingagents.agents.utils.agent_utils import get_language_instruction

def create_bull_researcher(llm):
    def bull_node(state) -> dict:
        investment_debate_state = state["investment_debate_state"]
        history = investment_debate_state.get("history", "")
        bull_history = investment_debate_state.get("bull_history", "")

        current_response = investment_debate_state.get("current_response", "")
        market_research_report = state["market_report"]
        sentiment_report = state["sentiment_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]

        prompt = f"""Bạn là nhà phân tích theo hướng “Bull” (ủng hộ đầu tư vào cổ phiếu). Nhiệm vụ là xây dựng một luận điểm thuyết phục, dựa trên bằng chứng, nhấn mạnh: tiềm năng tăng trưởng, lợi thế cạnh tranh, và các tín hiệu tích cực trên thị trường.{get_language_instruction()}

Yêu cầu nội dung:
- **Tiềm năng tăng trưởng**: cơ hội thị trường, dự phóng doanh thu/lợi nhuận (nếu có), khả năng mở rộng.
- **Lợi thế cạnh tranh**: sản phẩm/dịch vụ, thương hiệu, vị thế ngành, lợi thế chi phí, mạng lưới phân phối, v.v.
- **Tín hiệu tích cực**: sức khoẻ tài chính, xu hướng ngành, tin tức tích cực gần đây, diễn biến kỹ thuật (nếu có dữ liệu).
- **Phản biện Bear**: phản bác trực diện các luận điểm của Bear bằng số liệu/chi tiết cụ thể, chỉ ra chỗ giả định quá bi quan hoặc thiếu chứng cứ.
- **Cách trình bày**: viết theo phong cách tranh luận, đối thoại trực tiếp với Bear (không chỉ liệt kê gạch đầu dòng).

Tài nguyên có sẵn:
- Market research report: {market_research_report}
- Social media sentiment report: {sentiment_report}
- Latest world affairs news: {news_report}
- Company fundamentals report: {fundamentals_report}
- Lịch sử tranh luận: {history}
- Luận điểm Bear gần nhất: {current_response}

Hãy dùng các tài nguyên trên để đưa ra lập luận Bull, phản biện Bear, và thể hiện một cuộc tranh luận “có chất” (lý lẽ rõ, có bằng chứng).
"""

        response = llm.invoke(prompt)

        argument = f"Bull Analyst: {response.content}"

        new_investment_debate_state = {
            "history": history + "\n" + argument,
            "bull_history": bull_history + "\n" + argument,
            "bear_history": investment_debate_state.get("bear_history", ""),
            "current_response": argument,
            "count": investment_debate_state["count"] + 1,
        }

        return {"investment_debate_state": new_investment_debate_state}

    return bull_node
