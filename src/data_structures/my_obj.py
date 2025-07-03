class MyObj:
    def __init__(self, obj: object, obj_name: str):
        self._obj = obj
        self._obj_name = obj_name

    def __getattr__(self, name):
        return getattr(self._obj, name)
    
    def __call__(self, *args, **kwargs):
        return MyObj(self._obj(*args, **kwargs), self._obj_name)
    
    @property
    def obj_name(self):
        return self._obj_name