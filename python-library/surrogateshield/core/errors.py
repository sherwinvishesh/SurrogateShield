"""Exceptions shared by the detection pipeline, the library and the app."""


class DetectorUnavailable(RuntimeError):
    """A detection stage that is switched on could not run (model or package
    missing, or the model failed on this input).

    Detection fails closed (audit I17): the text is not masked and must not
    be sent. Install what the message names, or switch the stage off
    explicitly in configuration.
    """
