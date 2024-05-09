import bittensor as bt
from models import base_models
from operation_logic import utils as operation_utils
from models import utility_models


POST_ENDPOINT = "predictions"


async def text_to_speech_clone_logic(
    body: base_models.TextToSpeechCloneIncoming, mock = False
) -> base_models.TextToSpeechCloneOutgoing:
    """Add gpu potential"""

    output = base_models.TextToSpeechCloneOutgoing()

    speech_response_body = None 
    if not mock:
        try:
            speech_response_body = await operation_utils.get_speech_clone_from_server(body, POST_ENDPOINT, timeout=15)
        except Exception as e:
            bt.logging.debug(f"An error occurred while getting speech clone from server: {str(e)}")
    else:
        print(f"Mocking the response for TextToSpeechClone!")
        speech_response_body = utility_models.AudioResponseBody(output="data:audio/wav;base64,get_mogged")
    
    # If safe for work but still no images, something went wrong probably
    if speech_response_body is None or speech_response_body.output is None:
        output.error_message = "Some error from the generation :/"
        return output

    bt.logging.info("✅ Generated speech from given text and reference ✨")
    output.audio_b64 = speech_response_body.output

    bt.logging.info(f"✅✨: output: {output.audio_b64}")

    #TODO (priority 2): add ImageBind workers to the output
    # output.imagebind_embeddings = speech_response_body.clip_embeddings

    return output