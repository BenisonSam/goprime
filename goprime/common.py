class static_property:
    """
    Same as property(), but passes obj.__class__ instead of obj to fget/fset/fdel.
    Original code for property emulation:
    https://docs.python.org/3.5/howto/descriptor.html#properties
    """

    def __init__(self, fget=None, fset=None, fdel=None, doc=None):
        self.fget = fget
        self.fset = fset
        self.fdel = fdel
        if doc is None and fget is not None:
            doc = fget.__doc__
        self.__doc__ = doc

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        if self.fget is None:
            raise AttributeError("unreadable attribute")
        return self.fget(obj.__class__)

    def __set__(self, obj, value):
        if self.fset is None:
            raise AttributeError("can't set attribute")
        self.fset(obj.__class__, value)

    def __delete__(self, obj):
        if self.fdel is None:
            raise AttributeError("can't delete attribute")
        self.fdel(obj.__class__)

    def getter(self, fget):
        return type(self)(fget, self.fset, self.fdel, self.__doc__)

    def setter(self, fset):
        return type(self)(self.fget, fset, self.fdel, self.__doc__)

    def deleter(self, fdel):
        return type(self)(self.fget, self.fset, fdel, self.__doc__)


# noinspection PyCompatibility
def static_class(cls):
    """
    Class decorator to add metaclass to our class.
    Metaclass uses to add descriptors to class attributes, see:
    http://stackoverflow.com/a/26634248/1113207
    """

    class Meta(type(cls)):
        pass

    for name, obj in vars(cls).items():
        if isinstance(obj, static_property):
            setattr(Meta, name, property(obj.fget, obj.fset, obj.fdel))

    class Wrapper(cls, metaclass=Meta):
        pass

    return Wrapper


def create_batches(list_obj, batch_size):
    # For item i in a range that is a length of list_obj,
    for i in range(0, len(list_obj), batch_size):
        # Create an index range for list_obj of batch_size items:
        yield list_obj[i:i + batch_size]
