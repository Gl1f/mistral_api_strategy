from typing import List, Dict, Optional, Any
from config import TOKEN
import requests
import base64
from ABC import abstractmethod

class RequestStrategy:
    @abstractmethod
    def execute(self, text: str, model: str, history: list = None, image_path: str = None) -> dict:
        pass

class TextRequestStrategy(RequestStrategy):
    def execute(self, text: str, model: str, history: list = None, image_path: str = None) -> dict:
        pass

class ImageRequestStrategy(RequestStrategy):
    def execute(self, text: str, model: str, history: list = None, image_path: str = None) -> dict:
        pass

class ChatFacade:
    def __init__(self, api_key: str) -> None:
        pass
    
    def change_strategy(self, strategy_type: str) -> None:
        pass
    
    def ask_question(self, text: str, model: str, image_path: str = None) -> dict:
        pass
    
    def get_history(self) -> list[tuple[str, dict]]:
        pass

    def clear_history(self) -> None:
        pass