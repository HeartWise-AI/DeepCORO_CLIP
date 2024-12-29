from typing import Dict, Type, Any, Optional


class RegistryError(Exception):
    """Base exception for registry-related errors."""
    pass


class Registry:
    """Base registry for managing different types of classes."""
    
    _registries: Dict[str, Dict[str, Type[Any]]] = {}
    
    @classmethod
    def register(
        cls, 
        registry_type: str
    ):
        """Create a registry for a specific type of class."""
        if not registry_type or not isinstance(registry_type, str):
            raise RegistryError("Registry type must be a non-empty string")

        def decorator(name: str):
            if not name or not isinstance(name, str):
                raise RegistryError("Registration name must be a non-empty string")

            def wrapper(class_type: Type[Any]) -> Type[Any]:
                if registry_type not in cls._registries:
                    cls._registries[registry_type] = {}
                
                # Check if name is already registered
                if name in cls._registries[registry_type]:
                    raise RegistryError(
                        f"Name '{name}' is already registered in '{registry_type}' registry"
                    )
                
                cls._registries[registry_type][name] = class_type
                class_type.name = name
                return class_type
            return wrapper
        return decorator
    
    @classmethod
    def get(
        cls, 
        registry_type: str, 
        name: str
    ) -> Optional[Type[Any]]:
        """Get a registered class by its registry type and name."""
        if registry_type not in cls._registries:
            raise RegistryError(f"Registry type '{registry_type}' does not exist")
        
        class_type = cls._registries[registry_type].get(name)
        if class_type is None:
            raise RegistryError(
                f"No class named '{name}' found in '{registry_type}' registry"
            )
        return class_type
    
    @classmethod
    def list_registered(
        cls, 
        registry_type: str
    ) -> Dict[str, Type[Any]]:
        """List all registered classes for a specific registry."""
        if registry_type not in cls._registries:
            raise RegistryError(f"Registry type '{registry_type}' does not exist")
        return cls._registries[registry_type].copy()  # Return a copy for safety

# Create specific decorators for different types
def register_runner(name: str):
    return Registry.register('runner')(name)

def register_model(name: str):
    return Registry.register('model')(name)

def register_project(name: str):
    return Registry.register('project')(name)

def register_config(name: str):
    return Registry.register('config')(name)