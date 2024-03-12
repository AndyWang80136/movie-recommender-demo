from typing import Any, List, Optional


class InstanceFactoryMixin:
    """Mixin class for creating instance
    """
    INSTANCE_CLASS: Optional[object] = None

    def create(self, **kwargs):
        return self.INSTANCE_CLASS(**kwargs)


class InstanceMixin:
    """Mixin class for instance operation
    """

    def _get(self, obj: Any, attr_list: List[str]):
        if not isinstance(obj, dict) or not attr_list:
            return obj
        else:
            attr = attr_list.pop(0)
            return self._get(obj[attr], attr_list)

    def get(self, attr_name: str):
        attr, *attr_list = attr_name.split('/')
        attr_obj = getattr(self, attr)
        return self._get(attr_obj, attr_list) if attr_list else attr_obj
