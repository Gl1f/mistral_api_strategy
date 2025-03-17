from config import TOKEN
import requests
import base64
from abc import abstractmethod, ABC
from typing import List, Dict, Optional, Any


class RequestStrategy(ABC):
    """
    Абстрактный базовый класс, определяющий интерфейс для стратегий запросов к API Mistral.
    """

    @abstractmethod
    def execute(
        self,
        text: str,
        model: str,
        history: Optional[List[Dict[str, Any]]] = None,
        image_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Абстрактный метод для выполнения запроса к API.

        Args:
            text: Текст запроса пользователя.
            model: Название модели Mistral для использования.
            history: История предыдущих сообщений (опционально).
            image_path: Путь к изображению для мультимодальных запросов (опционально).

        Returns:
            Ответ от API в формате словаря.
        """
        pass


class TextRequestStrategy(RequestStrategy):
    """
    Стратегия для отправки текстовых запросов к API Mistral.
    """

    def __init__(self, api_key: str):
        """
        Инициализация стратегии текстовых запросов.

        Args:
            api_key: Ключ API для доступа к Mistral API.
        """
        self.api_key: str = api_key
        self.url: str = "https://api.mistral.ai/v1/chat/completions"
        self.headers: Dict[str, str] = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        self.history: List[Dict[str, Any]] = []

    def execute(
        self,
        text: str,
        model: str,
        history: Optional[List[Dict[str, Any]]] = None,
        image_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Выполняет текстовый запрос к API Mistral.

        Args:
            text: Текст запроса пользователя.
            model: Название модели Mistral для использования.
            history: История предыдущих сообщений (опционально).
            image_path: Не используется в текстовой стратегии, но требуется для соответствия интерфейсу.

        Returns:
            Ответ от API в формате словаря или строка с ошибкой.
        """
        self.text: str = text
        self.model: str = model
        self.history: Optional[List[Dict[str, Any]]] = history

        self.image_path: Optional[str] = image_path
        if history is None:
            self.history = [{"role": "user", "content": self.text}]
        else:
            self.history.append({"role": "user", "content": self.text})
        self.data: Dict[str, Any] = {
            "model": f"{self.model}",
            "messages": self.history,
        }
        self.response: requests.Response = requests.post(
            self.url, headers=self.headers, json=self.data
        )
        if self.response.status_code == 200:
            self.history.append(
                {
                    "role": "assistant",
                    "content": self.response.json()["choices"][0]["message"]["content"],
                }
            )
            return self.response.json()
        else:
            return f"Error: {self.response.status_code}"


class ImageRequestStrategy(RequestStrategy):
    """
    Стратегия для отправки мультимодальных запросов (текст + изображение) к API Mistral.
    """

    def __init__(self, api_key: str):
        """
        Инициализация стратегии мультимодальных запросов.

        Args:
            api_key: Ключ API для доступа к Mistral API.
        """
        self.api_key: str = api_key
        self.url: str = "https://api.mistral.ai/v1/chat/completions"
        self.headers: Dict[str, str] = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        self.history: List[Dict[str, Any]] = []

    def execute(
        self,
        text: str,
        model: str,
        history: Optional[List[Dict[str, Any]]] = None,
        image_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Выполняет мультимодальный запрос (текст + изображение) к API Mistral.

        Args:
            text: Текст запроса пользователя.
            model: Название модели Mistral для использования.
            history: История предыдущих сообщений (опционально).
            image_path: Путь к изображению для отправки вместе с запросом.

        Returns:
            Ответ от API в формате словаря или строка с ошибкой.

        Raises:
            Exception: Если указанный файл изображения не найден.
        """
        self.text: str = text
        self.model: str = model
        self.history: Optional[List[Dict[str, Any]]] = history
        self.image_path: Optional[str] = image_path
        self.image_data: str = ""

        if self.image_path:
            try:
                with open(self.image_path, "rb") as image_file:
                    self.image_data = base64.b64encode(image_file.read()).decode(
                        "utf-8"
                    )
                    self.image_data = f"data:image/jpeg;base64,{self.image_data}"
            except FileNotFoundError:
                raise Exception(f"Image file not found: {self.image_path}")
        if history is None:
            self.history = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self.text},
                        {"type": "image_url", "image_url": self.image_data},
                    ],
                }
            ]
        else:
            self.history.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self.text},
                        {"type": "image_url", "image_url": self.image_data},
                    ],
                }
            )
        self.data: Dict[str, Any] = {"model": f"{self.model}", "messages": self.history}
        self.response: requests.Response = requests.post(
            self.url, headers=self.headers, json=self.data
        )
        if self.response.status_code == 200:
            self.history.append(
                {
                    "role": "assistant",
                    "content": self.response.json()["choices"][0]["message"]["content"],
                }
            )
            return self.response.json()
        else:
            return f"Error: {self.response.status_code}"


class ChatFacade:
    """
    Фасад для упрощения взаимодействия с API Mistral, скрывающий детали реализации стратегий.
    """

    def __init__(self, api_key: str) -> None:
        """
        Инициализация фасада чата.

        Args:
            api_key: Ключ API для доступа к Mistral API.
        """
        self.api_key: str = api_key
        self.text_request: TextRequestStrategy = TextRequestStrategy(api_key)
        self.image_request: ImageRequestStrategy = ImageRequestStrategy(api_key)
        self.request_strategy: Optional[RequestStrategy] = None
        self.models_text: List[str] = [
            "mistral-small-latest",
            "open-mistral-nemo",
            "open-codestral-mamba",
        ]
        self.models_image: List[str] = [
            "pixtral-12b-2409",
        ]

    def change_strategy(self, strategy_type: str) -> None:
        """
        Изменяет текущую стратегию запросов.

        Args:
            strategy_type: Тип стратегии ('text' или 'image').

        Raises:
            ValueError: Если указан неверный тип стратегии.
        """
        if strategy_type == "text":
            self.request_strategy = self.text_request
        elif strategy_type == "image":
            self.request_strategy = self.image_request
        else:
            raise ValueError("Invalid strategy type")

    def select_model(self) -> str:
        """
        Предоставляет пользователю выбор модели в зависимости от текущей стратегии.

        Returns:
            Название выбранной модели.

        Raises:
            ValueError: Если указан неверный номер модели или неверный тип стратегии.
        """
        if self.request_strategy == self.text_request:
            for num in range(len(self.models_text)):
                print(f"{num + 1}. {self.models_text[num]}")
            l_num: int = int(input("Выберите модель: "))
            if l_num in range(1, len(self.models_text) + 1):
                return self.models_text[l_num - 1]
            else:
                raise ValueError("Invalid model number")
        elif self.request_strategy == self.image_request:
            for num in range(len(self.models_image)):
                print(f"{num + 1}. {self.models_image[num]}")
            l_num: int = int(input("Выберите модель: "))
            if l_num in range(1, len(self.models_image) + 1):
                return self.models_image[l_num - 1]
            else:
                raise ValueError("Invalid model number")
        else:
            raise ValueError("Invalid strategy type")

    def ask_question(
        self, text: str, model: str, image_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Отправляет запрос к API Mistral с использованием текущей стратегии.

        Args:
            text: Текст запроса пользователя.
            model: Название модели Mistral для использования.
            image_path: Путь к изображению для мультимодальных запросов (опционально).

        Returns:
            Ответ от API в формате словаря.

        Raises:
            ValueError: Если текущая стратегия не установлена или неверна.
        """
        if self.request_strategy == self.text_request:
            return self.text_request.execute(text, model, image_path=image_path)
        elif self.request_strategy == self.image_request:
            return self.image_request.execute(text, model, image_path=image_path)
        else:
            raise ValueError("Invalid strategy type")

    def get_history(self) -> List[Dict[str, Any]]:
        """
        Возвращает историю сообщений текущей стратегии.

        Returns:
            Список сообщений в истории.

        Raises:
            ValueError: Если текущая стратегия не установлена или неверна.
        """
        if self.request_strategy == self.text_request:
            return self.text_request.history
        elif self.request_strategy == self.image_request:
            return self.image_request.history
        else:
            raise ValueError("Invalid strategy type")

    def clear_history(self) -> None:
        """
        Очищает историю сообщений текущей стратегии.

        Raises:
            ValueError: Если текущая стратегия не установлена или неверна.
        """
        if self.request_strategy == self.text_request:
            self.text_request.history = []
        elif self.request_strategy == self.image_request:
            self.image_request.history = []
        else:
            raise ValueError("Invalid strategy type")


if __name__ == "__main__":
    api_key: str = TOKEN
    chat: ChatFacade = ChatFacade(api_key)

    # Смена стратегии
    chat.change_strategy("text")

    # Выбор модели
    model: str = chat.select_model()

    # Отправка текстового запроса
    текст_вопроса: str = "Расскажите о последних новостях в IT."
    response: Dict[str, Any] = chat.ask_question(текст_вопроса, model)

    print("Ответ от API:", response)

    # Смена стратегии на мультимодальную
    chat.change_strategy("image")

    # Выбор модели
    model: str = chat.select_model()

    # Отправка запроса с изображением
    текст_вопроса: str = "Расскажите что видишь на картинке."
    image_path: str = "castle.jpg"
    response: Dict[str, Any] = chat.ask_question(текст_вопроса, model, image_path)

    print("Ответ от API:", response)

    # Просмотр истории запросов
    print("История запросов:", chat.get_history())
    # Очистка истории запросов
    chat.clear_history()
    print("История запросов после очистки:", chat.get_history())
