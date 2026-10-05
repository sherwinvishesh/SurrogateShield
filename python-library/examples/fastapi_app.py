"""A masking service in front of any LLM, with FastAPI.

    pip install surrogateshield fastapi uvicorn
    python -m spacy download en_core_web_lg
    uvicorn fastapi_app:app            # from this directory

    POST   /mask                 {"text": "...", "conversation": null}
                                 → {"conversation": "<id>", "text": "<masked>"}
    POST   /unmask               {"conversation": "<id>", "text": "<LLM answer>"}
                                 → {"text": "<restored>"}
    DELETE /conversations/<id>   erase that conversation's map

The client sends the masked text to its LLM and passes the answer to
/unmask. Each conversation has one ``Session``, created by the server (ids
from clients are only looked up, never created), held in memory, and the
oldest is closed once there are more than ``max_conversations``. Responses
never contain the original values or the surrogate map.

Masking runs in a worker thread (``Session.amask``), so one slow request does
not stall the others. If a detection model is missing or fails, /mask returns
503 and no text: the service fails closed. The models load at startup, so a
broken install fails there and not on the first user's request.
"""

from collections import OrderedDict
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import surrogateshield as shield

MAX_TEXT = 100_000


class MaskIn(BaseModel):
    text: str = Field(max_length=MAX_TEXT)
    conversation: Optional[str] = None


class MaskOut(BaseModel):
    conversation: str
    text: str


class UnmaskIn(BaseModel):
    conversation: str
    text: str = Field(max_length=MAX_TEXT)


class TextOut(BaseModel):
    text: str


def make_app(config: Optional[shield.Config] = None, max_conversations: int = 1000) -> FastAPI:
    sessions: "OrderedDict[str, shield.Session]" = OrderedDict()

    def lookup(cid: str) -> shield.Session:
        s = sessions.get(cid)
        if s is None:
            raise HTTPException(404, "unknown conversation")
        sessions.move_to_end(cid)
        return s

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        async with shield.Session(config=config) as warm:
            await warm.ascan("warm-up: Dana lives in Tempe.")   # raises if a model is missing
        yield
        for s in sessions.values():
            s.close()
        sessions.clear()

    app = FastAPI(title="SurrogateShield masking service", lifespan=lifespan)

    @app.post("/mask", response_model=MaskOut)
    async def mask(body: MaskIn) -> MaskOut:
        if body.conversation is None:
            s = shield.Session(config=config)
            sessions[s.id] = s
            while len(sessions) > max_conversations:
                sessions.popitem(last=False)[1].close()
        else:
            s = lookup(body.conversation)
        try:
            text = await s.amask(body.text)
        except shield.DetectorUnavailable:
            raise HTTPException(503, "PII detection unavailable; nothing was masked") from None
        return MaskOut(conversation=s.id, text=text)

    @app.post("/unmask", response_model=TextOut)
    async def unmask(body: UnmaskIn) -> TextOut:
        return TextOut(text=await lookup(body.conversation).aunmask(body.text))

    @app.delete("/conversations/{cid}", status_code=204)
    async def forget(cid: str) -> None:
        lookup(cid)
        sessions.pop(cid).close()

    return app


app = make_app()
