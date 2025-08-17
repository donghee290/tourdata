def simple_sentiment(text: str) -> str:
    """간단한 감성 분석 (긍정/부정/중립)"""
    positive_keywords = ["좋다", "맛있", "즐겁", "만족", "추천"]
    negative_keywords = ["별로", "아쉽", "불편", "실망", "최악"]

    score = 0
    for w in positive_keywords:
        if w in text:
            score += 1
    for w in negative_keywords:
        if w in text:
            score -= 1

    if score > 0:
        return "긍정"
    elif score < 0:
        return "부정"
    else:
        return "중립"