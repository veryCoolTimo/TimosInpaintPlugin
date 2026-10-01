"""
Базовый класс для движков инпейнтинга
"""
from abc import ABC, abstractmethod
from typing import Callable, Optional
from PIL import Image


class BaseEngine(ABC):
    """
    Абстрактный базовый класс для движков инпейнтинга.

    Позволяет легко переключаться между:
    - Diffusers (MVP)
    - ComfyUI (headless)
    - CoreML (будущее)
    """

    # Наибольшая сторона, которую движок обрабатывает за один проход.
    # None = любое разрешение (кроп уходит в модель как есть). Больший кроп
    # main.py уменьшает до этого размера, а результат растягивает обратно
    # только внутри маски — пиксели вне маски всегда остаются оригинальными.
    max_resolution: Optional[int] = None
    # Размеры входа должны делиться на это число (8 для SD/LaMa, 16 для
    # FLUX.2 klein, 32 для FLUX.1 Fill)
    size_divisor: int = 8

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
        step_callback: Optional[Callable[[int, int], None]] = None,
        **kwargs,
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
            step_callback: вызывается как step_callback(step, total) между
                шагами денойзинга (используется для прогресса и отмены).
                Игнорируется движками без пошагового инференса (LaMa, OpenCV).
            **kwargs: место для параметров, специфичных для конкретного
                движка, чтобы вызывающий код (main.py) мог передавать их
                единообразно всем движкам, не проверяя тип каждый раз.

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
