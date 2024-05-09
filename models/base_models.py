"""
Models corresponding to MINER requests

The naming convention is super important to adhere too!

Keep it as SynapseNameBase / SynapseNameIncoming / SynapseNameOutgoing

Also, all of the fields should MUST have a default value (can be `None`, 0 etc) for the purpose of mock tests (scripts in `tests` folder)
"""

from typing import List, Optional
from pydantic import BaseModel, Field


from core import constants as cst
from core import dataclasses as dc
from models import utility_models
import bittensor as bt


class BaseSynapse(bt.Synapse):
    error_message: Optional[str] = Field(None)


class BaseOutgoing(BaseModel):
    error_message: Optional[str] = Field(None, description="The error message", title="error_message")


# AVAILABLE OPERATIONS


class AvailableTasksOperationIncoming(BaseModel): ...


class AvailableTasksOperationOutgoing(BaseModel):
    available_tasks: Optional[List[str]]


class AvailableTasksOperationBase(AvailableTasksOperationIncoming, AvailableTasksOperationOutgoing): ...


# Speech generation

class SpeechGenerationBase(BaseModel):
    alpha: float = Field(cst.DEFAULT_ALPHA ,description = "Only used for long text inputs or in case of reference speaker, determines the timbre of the speaker. Use lower values to sample style based on previous or reference speech instead of text.")
    beta: float = Field(cst.DEFAULT_BETA, description = "Only used for long text inputs or in case of reference speaker, determines the prosody of the speaker. Use lower values to sample style based on previous or reference speech instead of text.")
    diffusion_steps: int = Field(cst.DEFAULT_STEPS, description = "Number of diffusion steps to run.")
    embedding_scale: float = Field(cst.DEFAULT_EMB_SCALE, description = "Embedding scale, use higher values for pronounced emotion")
    seed: int = Field(cst.DEFAULT_SEED, description = "Seed for reproducibility")
    is_mock: bool = Field(False, description = "Set to True when running unit tests in `tests/mining/proxy` folder, in which case the requests do not go to the miner-worker")
    engine: utility_models.EngineEnum = Field(
        default=utility_models.EngineEnum.STYLETTS2.value, description="The engine to use for speech cloning and generation"
    )
    class Config:
        use_enum_values = True


class SpeechResponseBase(BaseOutgoing):
    audio_b64: str = Field(None, description="The base64 encoded audio file", title="audio_b64", alias = "output")

    class Config:
        populate_by_name = True
        allow_population_by_field_name = True  # Allow both aliases and field names
    #TODO (priority 2): add ImageBind workers to the output

class TextToSpeechCloneIncoming(SpeechGenerationBase):
    text: str = Field("This is a sample recording", description="The text to generate the speech from", title="text")
    reference: str = Field("", description="Reference speaker sample as base64 audio or downloadable URL", title="reference")


class TextToSpeechCloneOutgoing(SpeechResponseBase): ...


class TextToSpeechCloneBase(TextToSpeechCloneIncoming, TextToSpeechCloneOutgoing): ...


# Generic image gen


class ImageGenerationBase(BaseModel):
    cfg_scale: float = Field(cst.DEFAULT_CFG_SCALE, description="Scale for the configuration")
    steps: int = Field(cst.DEFAULT_TTS_STEPS, description="Number of steps in the image generation process")
    seed: int = Field(
        ...,
        description="Random seed for generating the image. NOTE: THIS CANNOT BE SET, YOU MUST PASS IN 0, SORRY!",
    )
    engine: utility_models.EngineEnum = Field(
        default=utility_models.EngineEnum.PROTEUS.value,
        description="The engine to use for image generation",
    )

    class Config:
        use_enum_values = True


class ImageResponseBase(BaseOutgoing):
    image_b64: Optional[str] = Field(None, description="The base64 encoded images to return", title="image_b64")
    clip_embeddings: Optional[List[float]] = Field(None, description="The clip embeddings for each of the images")
    image_hashes: Optional[utility_models.ImageHashes] = Field(None, description="Image hash's for each image")
    is_nsfw: Optional[bool] = Field(None, description="Is the image NSFW")


# TEXT TO IMAGE
class TextToImageIncoming(ImageGenerationBase):
    text_prompts: List[dc.TextPrompt] = Field([], description="Prompts for the image generation", title="text_prompts")

    height: int = Field(cst.DEFAULT_HEIGHT, description="Height of the generated image")
    width: int = Field(cst.DEFAULT_WIDTH, description="Width of the generated image")


class TextToImageOutgoing(ImageResponseBase): ...


class TextToImageBase(TextToImageIncoming, TextToImageOutgoing): ...


# IMAGE TO IMAGE


class ImageToImageIncoming(ImageGenerationBase):
    init_image: Optional[str] = Field(..., description="The base64 encoded image", title="init_image")
    text_prompts: List[dc.TextPrompt] = Field([], description="Prompts for the image generation", title="text_prompts")
    image_strength: float = Field(0.25, description="The strength of the init image")

    height: Optional[int] = Field(None, description="Height of the generated image")
    width: Optional[int] = Field(None, description="Width of the generated image")


class ImageToImageOutgoing(ImageResponseBase): ...


class ImageToImageBase(ImageToImageIncoming, ImageToImageOutgoing): ...


# Inpaint


class InpaintIncoming(BaseModel):
    seed: int = Field(..., description="Random seed for generating the image")
    steps: int = Field(8, description="Number of steps in the image generation process")
    cfg_scale: float = Field(3.0, description="Scale for the configuration")

    init_image: Optional[str] = Field(..., description="The base64 encoded image", title="init_image")
    text_prompts: List[dc.TextPrompt] = Field([], description="Prompts for the image generation", title="text_prompts")

    mask_image: Optional[str] = Field(None, description="The base64 encoded mask", title="mask_source")

    class Config:
        use_enum_values = True


class InpaintOutgoing(ImageResponseBase): ...


class InpaintBase(InpaintIncoming, InpaintOutgoing): ...


class AvatarIncoming(BaseModel):
    text_prompts: List[dc.TextPrompt] = Field(..., description="Prompts for the image generation", title="text_prompts")
    init_image: Optional[str] = Field(..., description="The base64 encoded image", title="image")
    ipadapter_strength: float = Field(..., description="The strength of the init image")
    control_strength: float = Field(..., description="The strength of the init image")
    height: int = Field(..., description="Height of the generated image")
    width: int = Field(..., description="Width of the generated image")
    steps: int = Field(..., description="Number of steps in the image generation process")
    seed: int = Field(
        ...,
        description="Random seed for generating the image. NOTE: THIS CANNOT BE SET, YOU MUST PASS IN 0, SORRY!",
    )

    class Config:
        use_enum_values = True


class AvatarOutgoing(ImageResponseBase): ...


class AvatarBase(AvatarIncoming, AvatarOutgoing): ...


# class ScribbleIncoming(ImageGenerationBase):
#     init_image: Optional[str] = Field(..., description="The base64 encoded image", title="init_image")
#     text_prompts: List[dc.TextPrompt] = Field([], description="Prompts for the image generation", title="text_prompts")

#     # Overriding defaults
#     engine: utility_models.EngineEnum = Field(utility_models.EngineEnum.sd, const=True)

#     image_strength: float = Field(0.25, description="The strength of the init image")

#     height: Optional[int] = Field(None, description="Height of the generated image")
#     width: Optional[int] = Field(None, description="Width of the generated image")


# class ScribbleOutgoing(ImageResponseBase): ...


# class ScribbleBase(ScribbleIncoming, ScribbleOutgoing): ...


# Upscale


class UpscaleIncoming(BaseModel):
    image: Optional[str] = Field(..., description="The base64 encoded image", title="image")


class UpscaleOutgoing(ImageResponseBase): ...


class UpscaleBase(UpscaleIncoming, UpscaleOutgoing): ...


# CLIP EMBEDDINGS
class ClipEmbeddingsIncoming(BaseModel):
    image_b64s: Optional[List[str]] = Field(
        None,
        description="The image b64s",
        title="image_b64s",
    )


class ClipEmbeddingsOutgoing(BaseOutgoing):
    clip_embeddings: Optional[List[List[float]]] = Field(
        None, description="The image clip embeddings", title="clip_embeddings"
    )


class ClipEmbeddingsBase(ClipEmbeddingsIncoming, ClipEmbeddingsOutgoing): ...


# SOTA
class SotaIncoming(BaseModel):
    prompt: str


class SotaOutgoing(BaseModel):
    image_url: Optional[str]
    error_message: Optional[str]


class SotaBase(SotaIncoming, SotaOutgoing): ...


class ChatIncoming(BaseModel):
    messages: list[utility_models.Message] = Field(...)

    temperature: float = Field(
        default=...,
        title="Temperature",
        description="Temperature for text generation.",
    )

    max_tokens: int = Field(500, title="Max Tokens", description="Max tokens for text generation.")

    seed: int = Field(
        default=...,
        title="Seed",
        description="Seed for text generation.",
    )

    # String as enums are really not playing nice with synapses
    model: str = Field(...)
    # model: utility_models.ChatModels = Field(default=..., title="Model")

    # @pydantic.validator("model", pre=True)
    # def validate_enum_field(cls, field):
    #     return utility_models.ChatModels(field)

    class Config:
        use_enum_values = True


class ChatOutgoing(BaseModel): ...


class ChatBase(ChatIncoming, ChatOutgoing): ...
