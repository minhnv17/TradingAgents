from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from tradingagents.agents.utils.agent_utils import (
    build_instrument_context,
    get_global_news,
    get_language_instruction,
    get_news,
)
from tradingagents.dataflows.config import get_config
from tradingagents.localization import get_agent_wrapper_message


def create_news_analyst(llm):
    def news_analyst_node(state):
        current_date = state["trade_date"]
        instrument_context = build_instrument_context(state["company_of_interest"])

        tools = [
            get_news,
            get_global_news,
        ]

        config = get_config()
        output_language = config.get("output_language", "English")
        agent_wrapper = get_agent_wrapper_message(output_language)

        system_message = (
            "Bạn là một nhà nghiên cứu tin tức phục vụ giao dịch, nhiệm vụ là phân tích tin tức và xu hướng trong 1 tuần gần đây."
            " Hãy viết một báo cáo đầy đủ về bối cảnh vĩ mô/thị trường và các sự kiện có liên quan đến giao dịch."
            " Dùng các công cụ có sẵn: `get_news(query, start_date, end_date)` để tìm tin tức theo công ty/chủ đề,"
            " và `get_global_news(curr_date, look_back_days, limit)` để lấy tin tức vĩ mô rộng hơn."
            " Nêu các nhận định cụ thể, có dẫn chứng, và chuyển hoá thành điểm hành động cho trader."
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
            "news_report": report,
        }

    return news_analyst_node
