import numpy as np
from typing import Dict, Type, Any
from filters.abstract_filters import AbstractFilters


class FiltersFactory:
    """Factory class to dynamically register and instantiate time-series filters."""

    # Class-level dictionary acting as the registry for available filters
    _registry: Dict[str, Type[AbstractFilters]] = {}

    @classmethod
    def register(cls, name: str):
        """Decorator to register a new filter class into the factory.

        Args:
            name (str): The unique string identifier for the filter.
        """

        def decorator(subclass: Type[AbstractFilters]):
            if not issubclass(subclass, AbstractFilters):
                raise TypeError(f"Class '{subclass.__name__}' must inherit from AbstractFilters.")
            cls._registry[name.lower()] = subclass
            return subclass

        return decorator

    @classmethod
    def create(cls, name: str, **kwargs: Any) -> AbstractFilters:
        """Instantiate and return a registered filter.

        Args:
            name (str): The unique string identifier of the filter.
            **kwargs: Initialization arguments for the filter's __init__.

        Returns:
            AbstractFilters: An instance of the requested filter.
        """
        lookup_name = name.lower()
        if lookup_name not in cls._registry:
            available_filters = ", ".join(cls._registry.keys())
            raise ValueError(
                f"Filter '{name}' is not registered. Available options: [{available_filters}]"
            )

        # Instantiate and return the matching filter class
        return cls._registry[lookup_name](**kwargs)
