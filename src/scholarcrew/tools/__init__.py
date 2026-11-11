from .arxiv import Paper, search_arxiv
from .reports import save_report
from .web import WebResult, search_web

__all__ = ["Paper", "WebResult", "save_report", "search_arxiv", "search_web"]
