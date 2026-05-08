"""Localization utilities for report generation in multiple languages."""

from typing import Dict


SECTION_HEADERS: Dict[str, Dict[str, str]] = {
    "en": {
        "title": "Trading Analysis Report: {ticker}",
        "generated": "Generated:",
        "analyst_team": "I. Analyst Team Reports",
        "research_team": "II. Research Team Decision",
        "trading_team": "III. Trading Team Plan",
        "risk_team": "IV. Risk Management Team Decision",
        "portfolio_team": "V. Portfolio Manager Decision",
    },
    "vi": {
        "title": "Báo Cáo Phân Tích Đầu Tư: {ticker}",
        "generated": "Ngày tạo:",
        "analyst_team": "I. Báo Cáo Đội Ngũ Phân Tích",
        "research_team": "II. Quyết Định Đội Ngũ Nghiên Cứu",
        "trading_team": "III. Kế Hoạch Đội Giao Dịch",
        "risk_team": "IV. Quyết Định Đội Ngũ Quản Lý Rủi Ro",
        "portfolio_team": "V. Quyết Định Đội Ngũ Quản Lý Danh Mục",
    },
}

ANALYST_NAMES: Dict[str, Dict[str, str]] = {
    "en": {
        "market_analyst": "Market Analyst",
        "sentiment_analyst": "Social Media Analyst",
        "news_analyst": "News Analyst",
        "fundamentals_analyst": "Fundamentals Analyst",
    },
    "vi": {
        "market_analyst": "Chuyên Gia Phân Tích Thị Trường",
        "sentiment_analyst": "Chuyên Gia Phân Tích Mạng Xã Hội",
        "news_analyst": "Chuyên Gia Phân Tích Tin Tức",
        "fundamentals_analyst": "Chuyên Gia Phân Tích Cơ Bản",
    },
}

RESEARCH_NAMES: Dict[str, Dict[str, str]] = {
    "en": {
        "bull": "Bull Researcher",
        "bear": "Bear Researcher",
        "manager": "Research Manager",
    },
    "vi": {
        "bull": "Nhà Phân Tích Tăng Giá",
        "bear": "Nhà Phân Tích Giảm Giá",
        "manager": "Quản Lý Nghiên Cứu",
    },
}

TRADER_NAMES: Dict[str, str] = {
    "en": "Trader",
    "vi": "Nhà Giao Dịch",
}

RISK_NAMES: Dict[str, Dict[str, str]] = {
    "en": {
        "aggressive": "Aggressive Analyst",
        "conservative": "Conservative Analyst",
        "neutral": "Neutral Analyst",
    },
    "vi": {
        "aggressive": "Chuyên Gia Phân Tích Tích Cực",
        "conservative": "Chuyên Gia Phân Tích Thận Trọng",
        "neutral": "Chuyên Gia Phân Tích Trung Lập",
    },
}

PORTFOLIO_NAMES: Dict[str, str] = {
    "en": "Portfolio Manager",
    "vi": "Quản Lý Danh Mục",
}

RATING_TRANSLATIONS: Dict[str, Dict[str, str]] = {
    "en": {
        "Buy": "Buy",
        "Overweight": "Overweight",
        "Hold": "Hold",
        "Underweight": "Underweight",
        "Sell": "Sell",
    },
    "vi": {
        "Buy": "Mua",
        "Overweight": "Tăng Tỷ Trọng",
        "Hold": "Nắm Giữ",
        "Underweight": "Giảm Tỷ Trọng",
        "Sell": "Bán",
    },
}


def get_language_code(language: str) -> str:
    """Convert language name to language code."""
    lang_lower = language.strip().lower()
    if lang_lower in {"vietnamese", "tiếng việt", "tieng viet", "vi"}:
        return "vi"
    return "en"


def get_section_headers(language: str) -> Dict[str, str]:
    """Get localized section headers."""
    code = get_language_code(language)
    return SECTION_HEADERS.get(code, SECTION_HEADERS["en"]).copy()


def get_analyst_name(role: str, language: str) -> str:
    """Get localized analyst role name."""
    code = get_language_code(language)
    return ANALYST_NAMES.get(code, ANALYST_NAMES["en"]).get(role, role)


def get_research_name(role: str, language: str) -> str:
    """Get localized research team member name."""
    code = get_language_code(language)
    return RESEARCH_NAMES.get(code, RESEARCH_NAMES["en"]).get(role, role)


def get_trader_name(language: str) -> str:
    """Get localized trader name."""
    code = get_language_code(language)
    return TRADER_NAMES.get(code, TRADER_NAMES["en"])


def get_risk_name(role: str, language: str) -> str:
    """Get localized risk analyst name."""
    code = get_language_code(language)
    return RISK_NAMES.get(code, RISK_NAMES["en"]).get(role, role)


def get_portfolio_name(language: str) -> str:
    """Get localized portfolio manager name."""
    code = get_language_code(language)
    return PORTFOLIO_NAMES.get(code, PORTFOLIO_NAMES["en"])


def get_rating_translation(rating: str, language: str) -> str:
    """Get localized rating."""
    code = get_language_code(language)
    return RATING_TRANSLATIONS.get(code, RATING_TRANSLATIONS["en"]).get(rating, rating)


def get_agent_wrapper_message(language: str) -> str:
    """Get localized agent wrapper message for system prompt."""
    code = get_language_code(language)

    wrappers = {
        "en": (
            "You are a helpful AI assistant, collaborating with other assistants. "
            "Use the provided tools to progress towards answering the question. "
            "If you are unable to fully answer, that's OK; another assistant with different tools "
            "will help where you left off. Execute what you can to make progress. "
            "If you or any other assistant has the FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** or deliverable, "
            "prefix your response with FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** so the team knows to stop. "
            "You have access to the following tools: {tool_names}.\n{system_message} "
            "For your reference, the current date is {current_date}. {instrument_context}"
        ),
        "vi": (
            "Bạn là một trợ lý AI, hợp tác với các trợ lý khác. "
            "Sử dụng các công cụ có sẵn để tiến hành trả lời câu hỏi. "
            "Nếu không thể trả lời đầy đủ thì cũng được; một trợ lý khác có công cụ khác sẽ giúp tiếp tục phần bạn để lại. "
            "Thực hiện những gì có thể để tiến triển. "
            "Nếu bạn hoặc bất kỳ trợ lý nào có ĐỀ XUẤT GIAO DỊCH CUỐI CÙNG: **MUA/NẮM GIỮ/BÁN** hoặc kết quả, "
            "hãy thêm tiền tố ĐỀ XUẤT GIAO DỊCH CUỐI CÙNG: **MUA/NẮM GIỮ/BÁN** để nhóm biết dừng lại. "
            "Bạn có quyền truy cập các công cụ sau: {tool_names}.\n{system_message} "
            "Để tham khảo, ngày hiện tại là {current_date}. {instrument_context}"
        ),
    }

    return wrappers.get(code, wrappers["en"])
