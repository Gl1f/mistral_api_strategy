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
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.url = 'https://api.mistral.ai/v1/chat/completions'
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
    
    def execute(
        self, text: str, model: str, history: list = None, image_path: str = None
    ) -> dict:
        self.text = text
        self.model = model
        self.history = history
        self.image_path = image_path
        if history is None:
            self.history = [{"role": "user", "content": self.text}]
        else:
            self.history.append({"role": "user", "content": self.text})
        self.data = {
            "model": f"{self.model}",
            "messages": self.history,
        }
        self.response = requests.post(self.url, headers=self.headers, json=self.data)
        print(self.history)


class ImageRequestStrategy(RequestStrategy):
    def __init__(self, api_key):
        self.api_key = api_key
        self.url = 'https://api.mistral.ai/v1/chat/completions'
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
    
    def execute(
        self, text: str, model: str, history: list = None, image_path: str = None
    ) -> dict:
        self.text = text
        self.model = model
        self.history = history
        self.image_path = image_path
        if self.image_path:
            try:
                with open(self.image_path, "rb") as image_file:
                    self.image_data = base64.b64encode(image_file.read()).decode("utf-8")
            except FileNotFoundError:
                raise Exception(f"Image file not found: {self.image_path}")
        
class ChatFacade:
    def __init__(self, api_key: str) -> None:
        self.api_key = api_key
        self.text_request = TextRequestStrategy(api_key)
        self.image_request = ImageRequestStrategy(api_key)
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
    api_key = TOKEN
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
