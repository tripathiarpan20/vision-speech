from fastapi import HTTPException
from models import base_models, synapses, utility_models, request_models
from validation.proxy import validation_utils
from starlette.responses import StreamingResponse
from core import tasks
from fastapi.routing import APIRouter
from validation.core_validator import get_validator_instance
import fastapi
from validation.proxy import dependencies
import bittensor as bt

router = APIRouter(tags=["speech"])


@router.post("/text-to-speech-clone")
async def speech(
    body: request_models.TextToSpeechCloneRequest,
    _: None = fastapi.Depends(dependencies.get_token),
) -> StreamingResponse:
    synapse = validation_utils.get_synapse_from_body(
        body=body,
        synapse_model=synapses.TextToSpeechClone,
    )

    if synapse.engine == utility_models.EngineEnum.STYLETTS2.value:
        task = tasks.Tasks.tts_clone.value
    else:
        raise HTTPException(status_code=400, detail="Invalid model provided")

    core_validator = get_validator_instance()
    result = await core_validator.execute_query(
        synapse, outgoing_model=base_models.TextToSpeechCloneOutgoing, task=tasks.Tasks.tts_clone.value
    )
    # bt.logging.info(f"!!!!!Result: {result}")

    if result is None:
        raise HTTPException(
            status_code=fastapi.status.HTTP_400_BAD_REQUEST,
            detail="I'm sorry, no valid response was possible from the miners :/",
        )

    bt.logging.info(f"!!!! Formatted Response: {result.formatted_response}")
    formatted_response: base_models.TextToSpeechCloneOutgoing = result.formatted_response


    return request_models.TextToSpeechCloneResponse(
        audio_b64 = formatted_response.audio_b64
    )
