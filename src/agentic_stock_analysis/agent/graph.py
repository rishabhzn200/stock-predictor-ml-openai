from langgraph.graph import StateGraph, START, END

from agentic_stock_analysis.agent.state import AgentState
from agentic_stock_analysis.agent.nodes.metadata import fetch_ticker_metadata_node
from agentic_stock_analysis.agent.nodes.plan_news_query import plan_news_query_node
from agentic_stock_analysis.agent.nodes.news import news_node
from agentic_stock_analysis.agent.nodes.news_sentiment import news_sentiment_node
from agentic_stock_analysis.agent.nodes.predict import predict_node
from agentic_stock_analysis.agent.nodes.alignment import alignment_node
from agentic_stock_analysis.agent.nodes.summarize import summarize_node


def build_agent_graph():
    graph = StateGraph(AgentState)

    graph.add_node("metadata", fetch_ticker_metadata_node)
    graph.add_node("plan_news_query", plan_news_query_node)
    graph.add_node("news", news_node)
    graph.add_node("news_sentiment", news_sentiment_node)
    graph.add_node("predict", predict_node)
    graph.add_node("alignment", alignment_node)
    graph.add_node("summarize", summarize_node)

    graph.add_edge(START, "metadata")
    graph.add_edge("metadata", "plan_news_query")
    graph.add_edge("plan_news_query", "news")
    graph.add_edge("news", "news_sentiment")
    graph.add_edge("news_sentiment", "predict")
    graph.add_edge("predict", "alignment")
    graph.add_edge("alignment", "summarize")
    graph.add_edge("summarize", END)

    return graph.compile()
