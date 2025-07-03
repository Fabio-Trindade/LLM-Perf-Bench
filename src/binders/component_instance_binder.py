class ComponentInstanceBinder:
    @staticmethod
    def create_registration_functions_from_catalog(catalog_cls):
        def decorator(registry_cls):
            def make_register_decorator(component_type, component_name):
                def register(instance_name):
                    def wrapper(instance):
                        registry_cls.register(component_type, component_name, instance_name, instance)
                        return instance
                    return wrapper

                register.__name__ = f"register_{component_name}_{component_type}"
                return register

            comp_typenames = catalog_cls.get_typenames()
            comp_names = catalog_cls.get_comp_names()

            for comp_typename in comp_typenames:
                for cname in comp_names:
                    func = make_register_decorator(comp_typename, cname)
                    setattr(registry_cls, func.__name__, staticmethod(func))

            return registry_cls

        return decorator
