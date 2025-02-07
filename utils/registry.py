from typing import Dict, Type, Any, Optional


class BaseRegistry:
    """Base registry class that implements common registry functionality."""
    _registry: Dict[str, Type[Any]] = {}
    _registry_type: str = "base"  # Should be overridden by subclasses
    
    @classmethod
    def register(cls, name: str):
        """Register a class in the registry."""
        def decorator(class_type: Type[Any]) -> Type[Any]:
            cls._registry[name] = class_type
            return class_type
        return decorator
    
    @classmethod
    def get(cls, name: str) -> Optional[Type[Any]]:
        """Get a registered class by name."""
        if name not in cls._registry:
            raise ValueError(
                f"{cls._registry_type} {name} not found in registry. "
                f"Available {cls._registry_type}s: {list(cls._registry.keys())}"
            )
        return cls._registry[name]
    
    @classmethod
    def list_registered(cls) -> Dict[str, Type[Any]]:
        """List all registered classes."""
        return cls._registry.copy()

    @classmethod
    def create(cls, name: str, **kwargs):
        """Create an instance of a registered class by name."""
        if name not in cls._registry:
            raise ValueError(
                f"{cls._registry_type} {name} not found in registry. "
                f"Available {cls._registry_type}s: {list(cls._registry.keys())}"
            )
        return cls._registry[name](**kwargs)


class RunnerRegistry(BaseRegistry):
    """Registry for runners."""
    _registry: Dict[str, Type[Any]] = {}
    _registry_type: str = "runner"

class ModelRegistry(BaseRegistry):
    """Registry for models."""
    _registry: Dict[str, Type[Any]] = {}
    _registry_type: str = "model"

class ProjectRegistry(BaseRegistry):
    """Registry for projects."""
    _registry: Dict[str, Type[Any]] = {}
    _registry_type: str = "project"

class ConfigRegistry(BaseRegistry):
    """Registry for configs."""
    _registry: Dict[str, Type[Any]] = {}
    _registry_type: str = "config"
