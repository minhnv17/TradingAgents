"""Trader: turns the Research Manager's investment plan into a concrete transaction proposal."""

from __future__ import annotations

import functools

from langchain_core.messages import AIMessage

from tradingagents.agents.schemas import TraderProposal, render_trader_proposal
from tradingagents.agents.utils.agent_utils import build_instrument_context, get_language_instruction
from tradingagents.agents.utils.structured import (
    bind_structured,
    invoke_structured_or_freetext,
)


def create_trader(llm):
    structured_llm = bind_structured(llm, TraderProposal, "Trader")

    def trader_node(state, name):
        company_name = state["company_of_interest"]
        instrument_context = build_instrument_context(company_name)
        investment_plan = state["investment_plan"]

        messages = [
            {
                "role": "system",
                "content": (
                    "Bạn là một tác nhân giao dịch (Trader) chịu trách nhiệm chuyển kế hoạch nghiên cứu thành đề xuất giao dịch cụ thể. "
                    "Dựa trên các báo cáo của Analyst và kế hoạch của Research Manager, hãy đưa ra khuyến nghị rõ ràng: mua/bán/giữ, "
                    "kèm luận cứ, rủi ro chính, và các điều kiện kích hoạt (nếu phù hợp)."
                    + get_language_instruction()
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Dưới đây là kế hoạch đầu tư do đội ngũ phân tích tổng hợp cho {company_name}. {instrument_context} "
                    f"Kế hoạch này kết hợp xu hướng kỹ thuật, vĩ mô và tâm lý thị trường. "
                    f"Hãy dùng nó làm nền tảng để đưa ra quyết định giao dịch tiếp theo.\n\n"
                    f"Kế hoạch đầu tư đề xuất: {investment_plan}\n\n"
                    f"Hãy biến các insight thành một đề xuất giao dịch mạch lạc và có thể thực thi."
                ),
            },
        ]

        trader_plan = invoke_structured_or_freetext(
            structured_llm,
            llm,
            messages,
            render_trader_proposal,
            "Trader",
        )

        return {
            "messages": [AIMessage(content=trader_plan)],
            "trader_investment_plan": trader_plan,
            "sender": name,
        }

    return functools.partial(trader_node, name="Trader")
