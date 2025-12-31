from agentic_stock_analysis.agent.state import AgentState


def alignment_node(state: AgentState) -> AgentState:
    pred = (state.get("prediction") or "").upper()
    label = (state.get("news_sentiment_label") or "").upper()

    # If no usable news
    if label in {"NO_NEWS"} or not pred:
        return {"alignment": "UNKNOWN"}

    # Map sentiment label to direction
    if label == "POSITIVE":
        news_dir = "UP"
    elif label == "NEGATIVE":
        news_dir = "DOWN"
    else:
        return {"alignment": "UNKNOWN"}

    return {"alignment": "ALIGNED" if news_dir == pred else "CONFLICT"}
