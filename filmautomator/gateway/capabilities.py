"""The capability vocabulary the Model Gateway routes on.

Spec section 2: the system must be able to determine which model is capable of
a requested operation, and must not assume the owner's model supports
everything. Every driver declares the set of capabilities it can serve; the
gateway refuses to route a request to a driver that does not claim it.
"""

from __future__ import annotations

from enum import Enum


class Capability(str, Enum):
    """A unit of model work the system may need performed."""

    REASONING = "reasoning"
    PLANNING = "planning"
    CODING = "coding"
    VISION = "vision"
    IMAGE_GENERATION = "image_generation"
    VIDEO_GENERATION = "video_generation"
    SPEECH_GENERATION = "speech_generation"
    TRANSCRIPTION = "transcription"
    MUSIC_GENERATION = "music_generation"
    SFX_GENERATION = "sfx_generation"
    EMBEDDINGS = "embeddings"

    def __str__(self) -> str:  # nicer log lines
        return self.value


#: Capabilities that require the model to accept image input.
VISION_INPUT_CAPABILITIES: frozenset[Capability] = frozenset({Capability.VISION})

#: Capabilities that are not yet wired to any free local driver. The Director
#: checks this set before assigning work, so an unimplemented capability
#: degrades gracefully instead of failing a task at runtime.
UNIMPLEMENTED_CAPABILITIES: frozenset[Capability] = frozenset(
    {
        Capability.VIDEO_GENERATION,
        Capability.MUSIC_GENERATION,
    }
)
