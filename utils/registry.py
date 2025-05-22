
import importlib
import pkgutil

from typing import (
    Dict, 
    Type, 
    Any, 
    Optional
)


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
                f"{cls._registry_type} {name} not found in registry {cls._registry_type}. "
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
                f"{cls._registry_type} {name} not found in registry {cls._registry_type}. "
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

class ParserRegistry(BaseRegistry):
    """Registry for parsers."""
    _registry: Dict[str, Type[Any]] = {}
    _registry_type: str = "parser"
    
class LossRegistry(BaseRegistry):
    """Registry for losses."""
    _registry: Dict[str, Type[Any]] = {}
    _registry_type: str = "loss"

def register_submodules(package_name: str, recursive: bool = True) -> None:
    """
    Dynamically import all submodules of a package.
    
    Args:
        package_name (str): Name of the package to import.
        recursive (bool): If True, recursively import submodules.
    """
    package = importlib.import_module(package_name)
    for _, name, is_pkg in pkgutil.walk_packages(package.__path__, package.__name__ + "."):
        importlib.import_module(name)
        if recursive and is_pkg:
            register_submodules(name, recursive=recursive)
