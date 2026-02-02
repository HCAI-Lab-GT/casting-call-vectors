from inspect_ai.model import modelapi


@modelapi(name="pvx")
def pvx():
    from .custom import PVXModelAPI

    return PVXModelAPI
