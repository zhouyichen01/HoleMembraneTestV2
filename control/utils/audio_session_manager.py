class AudioSessionManager:
    _current_owner = None

    @classmethod
    def acquire(cls, owner) -> bool:
        """owner 一般传 self（窗口对象）或一个字符串标识"""
        if cls._current_owner is not None:
            return False
        cls._current_owner = owner
        return True

    @classmethod
    def release(cls, owner):
        if cls._current_owner is owner:
            cls._current_owner = None