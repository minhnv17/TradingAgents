from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from tradingagents.agents.utils.agent_utils import (
    build_instrument_context,
    get_balance_sheet,
    get_cashflow,
    get_fundamentals,
    get_income_statement,
    get_insider_transactions,
    get_language_instruction,
)
from tradingagents.dataflows.config import get_config
from tradingagents.localization import get_agent_wrapper_message


def create_fundamentals_analyst(llm):
    def fundamentals_analyst_node(state):
        current_date = state["trade_date"]
        instrument_context = build_instrument_context(state["company_of_interest"])

        tools = [
            get_fundamentals,
            get_balance_sheet,
            get_cashflow,
            get_income_statement,
        ]

        config = get_config()
        output_language = config.get("output_language", "English")
        agent_wrapper = get_agent_wrapper_message(output_language)

        system_message = (
            "Bạn là một nhà nghiên cứu phân tích cơ bản (fundamental) cho giao dịch, nhiệm vụ là phân tích thông tin cơ bản của công ty trong 1 tuần gần đây."
            " Hãy viết một báo cáo đầy đủ về: hồ sơ công ty, các chỉ số tài chính cơ bản, lịch sử tài chính, và các điểm mạnh/yếu quan trọng"
            " để giúp trader ra quyết định."
            " Nêu nhận định cụ thể, có dẫn chứng từ dữ liệu, và chỉ ra tác động lên rủi ro/lợi nhuận kỳ vọng."
            + " Dùng các công cụ có sẵn: `get_fundamentals` (tổng quan), `get_balance_sheet` (BCĐKT), `get_cashflow` (LCTT), `get_income_statement` (KQKD)."
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
            "fundamentals_report": report,
        }

    return fundamentals_analyst_node
