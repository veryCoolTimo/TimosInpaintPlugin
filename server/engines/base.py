"""
Базовый класс для движков инпейнтинга
"""
from abc import ABC, abstractmethod
from typing import Optional
from PIL import Image


class BaseEngine(ABC):
    """
    Абстрактный базовый класс для движков инпейнтинга.

    Позволяет легко переключаться между:
    - Diffusers (MVP)
    - ComfyUI (headless)
    - CoreML (будущее)
    """

    @abstractmethod
    def load(self) -> None:
        """Загружает модель в память"""
        pass

    @abstractmethod
    def unload(self) -> None:
        """Выгружает модель из памяти"""
        pass

    @abstractmethod
    def is_loaded(self) -> bool:
        """Проверяет загружена ли модель"""
        pass

    @abstractmethod
    def inpaint(
        self,
        image: Image.Image,
        mask: Image.Image,
        prompt: str = "",
        negative_prompt: str = "",
        strength: float = 0.85,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 30,
        controlnet_scale: float = 0.5,
        seed: Optional[int] = None,
    ) -> Image.Image:
        """
        Выполняет инпейнтинг.

        Args:
            image: Исходное изображение (RGB)
            mask: Маска (L mode, белый = область инпейнта)
            prompt: Текстовый промпт
            negative_prompt: Негативный промпт
            strength: Сила деноизинга (0.0-1.0)
            guidance_scale: CFG scale
            num_inference_steps: Количество шагов
            controlnet_scale: Сила ControlNet (0.0-1.0)
            seed: Сид для воспроизводимости

        Returns:
            Результат инпейнтинга (RGB)
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Имя движка"""
        pass

    @property
    @abstractmethod
    def supports_controlnet(self) -> bool:
        """Поддерживает ли движок ControlNet"""
        pass
