from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from tradingagents.agents.utils.agent_utils import (
    build_instrument_context,
    get_language_instruction,
    get_news,
)
from tradingagents.dataflows.config import get_config
from tradingagents.localization import get_agent_wrapper_message


def create_social_media_analyst(llm):
    def social_media_analyst_node(state):
        current_date = state["trade_date"]
        instrument_context = build_instrument_context(state["company_of_interest"])

        tools = [
            get_news,
        ]

        config = get_config()
        output_language = config.get("output_language", "English")
        agent_wrapper = get_agent_wrapper_message(output_language)

        system_message = (
            "Bạn là một nhà phân tích mạng xã hội và tin tức doanh nghiệp, nhiệm vụ là đánh giá các thảo luận trên mạng xã hội,"
            " tin tức gần đây và tâm lý công chúng đối với một công ty trong 1 tuần qua."
            " Bạn sẽ nhận được tên/mã công ty; mục tiêu là viết một báo cáo dài, chi tiết về:"
            " (1) mọi người đang nói gì, (2) tâm lý/độ tích cực-tiêu cực theo thời gian, (3) tin tức doanh nghiệp và tác động."
            " Dùng công cụ `get_news(query, start_date, end_date)` để tìm các thảo luận/tin tức liên quan."
            " Cố gắng bao quát nhiều nguồn nhất có thể, và kết luận thành các điểm hành động cụ thể cho trader/nhà đầu tư."
            + " BẮT BUỘC thêm một bảng Markdown ở cuối báo cáo để tóm tắt các ý chính (rõ ràng, dễ đọc)."
            + get_language_instruction()
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    agent_wrapper,
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        prompt = prompt.partial(system_message=system_message)
        prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
        prompt = prompt.partial(current_date=current_date)
        prompt = prompt.partial(instrument_context=instrument_context)

        chain = prompt | llm.bind_tools(tools)

        result = chain.invoke(state["messages"])

        report = ""

        if len(result.tool_calls) == 0:
            report = result.content

        return {
            "messages": [result],
            "sentiment_report": report,
        }

    return social_media_analyst_node
