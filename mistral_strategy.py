from typing import List, Dict, Optional, Any
from config import TOKEN
import requests
import base64
from abc import abstractmethod, ABC


class RequestStrategy(ABC):
    @abstractmethod
    def execute(
        self, text: str, model: str, history: list = None, image_path: str = None
    ) -> dict:
        pass


class TextRequestStrategy(RequestStrategy):
    def execute(
        self, text: str, model: str, history: list = None, image_path: str = None
    ) -> dict:
        self.text = text
        self.model = model
        self.history = history
        self.image_path = image_path
        self.api_key = ChatFacade.api_key
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        self.url = 'https://api.mistral.ai/v1/chat/completions'
        self.data = {
            "model": f"{self.model}",
            "messages": self.history,
        }
        self.response = requests.post(self.url, headers=self.headers, json=self.data)


class ImageRequestStrategy(RequestStrategy):
    def execute(
        self, text: str, model: str, history: list = None, image_path: str = None
    ) -> dict:
        self.text = text
        self.model = model
        self.history = history
        self.image_path = image_path
        self.url = ChatFacade.url
        self.headers = ChatFacade.headers


class ChatFacade:
    def __init__(self, api_key: str) -> None:
        self.api_key = api_key
        self.url: str = "https://api.mistral.ai/v1/chat/completions"
        self.headers: Dict[str, str] = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        self.text_request = TextRequestStrategy()
        self.image_request = ImageRequestStrategy()
        self.models_text = [
            "mistral-small-latest",
            "open-mistral-nemo",
            "open-codestral-mamba",
        ]
        self.models_image = [
            "pixtral-12b-2409",
        ]

    def change_strategy(self, strategy_type: str) -> None:
        if strategy_type == "text":
            self.request_strategy = self.text_request
        elif strategy_type == "image":
            self.request_strategy = self.image_request
        else:
            raise ValueError("Invalid strategy type")

    def select_model(self) -> str:
        if self.request_strategy == self.text_request:
            for num in range(len(self.models_text)):
                print(f"{num + 1}. {self.models_text[num]}")
            l_num = int(input("Выберите модель: "))
            if l_num in range(1, len(self.models_text) + 1):
                return self.models_text[l_num - 1]
            else:
                raise ValueError("Invalid model number")
        elif self.request_strategy == self.image_request:
            for num in range(len(self.models_image)):
                print(f"{num + 1}. {self.models_image[num]}")
            l_num = int(input("Выберите модель: "))
            if l_num in range(1, len(self.models_image) + 1):
                return self.models_image[l_num - 1]
            else:
                raise ValueError("Invalid model number")
        else:
            raise ValueError("Invalid strategy type")

    def ask_question(self, text: str, model: str, image_path: str = None) -> dict:
        if self.request_strategy == self.text_request:
            return self.text_request.execute(text, model, image_path=image_path)
        elif self.request_strategy == self.image_request:
            return self.image_request.execute(text, model, image_path=image_path)
        else:
            raise ValueError("Invalid strategy type")

    def get_history(self) -> list[tuple[str, dict]]:
        if self.request_strategy == self.text_request:
            return self.text_request.history
        elif self.request_strategy == self.image_request:
            return self.image_request.history
        else:
            raise ValueError("Invalid strategy type")

    def clear_history(self) -> None:
        if self.request_strategy == self.text_request:
            self.text_request.history = []
        elif self.request_strategy == self.image_request:
            self.image_request.history = []
        else:
            raise ValueError("Invalid strategy type")

if __name__ == "__main__":
    api_key = "your_api_key_here"
    chat = ChatFacade(api_key)

    # Смена стратегии
    chat.change_strategy("text")

    # Выбор модели
    model = chat.select_model()

    # Отправка текстового запроса
    текст_вопроса = "Расскажите о последних новостях в IT."
    response = chat.ask_question(текст_вопроса, model)

    print("Ответ от API:", response)

    # Смена стратегии на мультимодальную
    chat.change_strategy("image")

    # Выбор модели
    model = chat.select_model()

    # Отправка запроса с изображением
    image_path = "path/to/image.jpg"
    response = chat.ask_question(текст_вопроса, model, image_path)

    print("Ответ от API:", response)

    # Просмотр истории запросов
    print("История запросов:", chat.get_history())
