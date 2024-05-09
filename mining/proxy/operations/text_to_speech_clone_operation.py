from typing import Tuple, TypeVar

import bittensor as bt

from mining.proxy import core_miner
from mining.proxy.operations import abstract_operation
from models import base_models, utility_models, synapses
from operation_logic import text_to_speech_clone_logic

operation_name = "TextToSpeechCloneOperation"

T = TypeVar("T", bound=bt.Synapse)


class TextToSpeechCloneOperation(abstract_operation.Operation):
    #TODO (priority 2): Add support for multiple engines using `utility_models.EngineEnum` (change `base_models.py` accordingly)
    @staticmethod
    async def forward(synapse: synapses.TextToSpeechClone) -> synapses.TextToSpeechClone:
        if synapse.engine is utility_models.EngineEnum.STYLETTS2.value:
            output = await text_to_speech_clone_logic.text_to_speech_clone_logic(base_models.TextToSpeechCloneIncoming(**synapse.dict()), mock = synapse.is_mock)
        #TODO (priority 2): Handle cases for other engines than StyleTTS2 (if any in future)
        output_dict = output.dict()
        for field in output_dict:
            setattr(synapse, field, output_dict[field])

        return synapse

    @staticmethod
    def blacklist(synapse: synapses.TextToSpeechClone) -> Tuple[bool, str]:
        return core_miner.base_blacklist(synapse)

    @staticmethod
    def priority(synapse: synapses.TextToSpeechClone) -> float:
        return core_miner.base_priority(synapse)