from fastapi import Request
from fastapi.responses import JSONResponse
from .base_exception import BaseException

async def common_exception_handler(request: Request, exc: BaseException):
    return JSONResponse(
        status_code = exc.status_code,
        content = exc.response.model_dump())