
class Version(object):
    """Version of the package"""

    def __setattr__(self, *args):
        raise TypeError("can't modify immutable instance")

    def __init__(self, num):
        super(Version, self).__setattr__('number', num)
    
    __delattr__ = __setattr__