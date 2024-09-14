from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from load_model import start_update_thread
import queue
import uvicorn
import logging


logging.basicConfig(level=logging.INFO)

_queue = queue.Queue()
output_queue = queue.Queue()

app = FastAPI()

start_update_thread(_queue, output_queue)


class QuestionRequest(BaseModel):
    question: str


@app.post("/ask/")
async def ask_question(request: QuestionRequest):
    try:
        _queue.put(request.question)
        response, confidence = output_queue.get()
        confidence = float(confidence)
        if confidence > 0.5:
            return JSONResponse(
                content={"response": response, "confidence": confidence},
                status_code=201,
            )
        else:
            return JSONResponse(
                content={"response": "Не уверен...", "confidence": confidence},
                status_code=202,
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9092)
