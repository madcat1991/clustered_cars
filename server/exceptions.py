class BaseApiException(Exception):
    """Base api exception. Other api exception classes have to be derived from it.
    """
    _default_message = u'unexpected problem has occurred'
    code = 0

    def __init__(self, message=None):
        self.message = message if message is not None else self._default_message

    def __str__(self):
        return u'<CODE {0.code}> {0.message}'.format(self)

    def as_dict(self):
        if self.code is not None:
            return {"error": self.message, "error_code": self.code}
        return {"error": self.message}


class ArgErrorException(BaseApiException):
    """Exceptions related to api query args
    """
    _default_message = u'arg is invalid'
    code = 400

    def __init__(self, arg_name, message=None, **kwargs):
        self.arg_name = arg_name
        self.kwargs = kwargs
        super(ArgErrorException, self).__init__(message)

    def as_dict(self):
        d = {"error": self.message, "error_code": self.code, "arg": self.arg_name}
        d.update(self.kwargs)
        return d
