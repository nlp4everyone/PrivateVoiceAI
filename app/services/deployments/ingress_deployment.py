# FastAPI components
from fastapi import UploadFile, File, Form, FastAPI
# Ray Serve for deployment
from ray import serve
from ray.serve.handle import DeploymentHandle
# Audio utilities
from app.utils.audio import (load_audio_from_bytes,
                             estimate_audio_duration,
                             is_audio_file)
from app.utils.transcription.helper import get_transcription_type
from app.utils.token_counter import approximate_count_tokens
from app.utils.language_detect import LanguageDetector
# Schema
from app.schema.segment import VADSegment
from app.schema.transcription.response import *
from app.schema.transcription import AdvancedTranscribedSegment
from app.schema.transcription.type import TranscriptionType
from app.schema.transcription.base.usage import *
# Custom exceptions
from app.exceptions.transcription import TranscriptedModelNotFoundException
from app.exceptions.audio import UnsupportedAudioFormatException
from app.exceptions.handlers import common_exception_handler
# Configuration imports
from app.core.config.serving import *
from app.core.config.asr import *
# Logging
import logging
# Other components
import torch, asyncio, math
from pathlib import Path

logger = logging.getLogger("ray.serve")

# Define tags metadata for API documentation
tags_metadata = [
    {
        "name": "Audio",
        "description": "Endpoints for interacting with ASR models"
    },
]

# Initialize FastAPI application for ingress service
ingress_app = FastAPI(openapi_tags=tags_metadata)

# Register custom exception handler
ingress_app.add_exception_handler(TranscriptedModelNotFoundException, common_exception_handler)
ingress_app.add_exception_handler(UnsupportedAudioFormatException, common_exception_handler)


@serve.deployment
class Pipeline:
    """
    Pipeline deployment that chains VAD and ASR deployments.

    This deployment coordinates the VAD+ASR pipeline:
    1. VAD detects speech segments in audio
    2. Each segment is transcribed by ASR
    3. Results are combined and returned
    """

    def __init__(self, vad: DeploymentHandle, asr: DeploymentHandle):
        """
        Initialize the pipeline with VAD and ASR deployment handles.

        Args:
            vad: VAD deployment handle
            asr: ASR deployment handle
        """
        self.vad = vad
        self.asr = asr

    def _extract_audio_segments(self, 
                                audio_tensor: torch.Tensor,
                                vad_segments: List[VADSegment],
                                sample_rate: int = 16000) -> List[torch.Tensor]:
        """
        Extract audio segments based on VAD timestamps.

        Args:
            audio_tensor: Full audio waveform as tensor
            vad_segments: List of VAD segments with start/end timestamps
            sample_rate: Sample rate of the audio (default: 16000)

        Returns:
            List of audio tensors for each speech segment
        """
        segments = []
        for vad_seg in vad_segments:
            # Convert timestamps to sample indices
            start_idx = int(vad_seg.start * sample_rate)
            end_idx = int(vad_seg.end * sample_rate)
            
            # Ensure indices are within bounds
            start_idx = max(0, start_idx)
            end_idx = min(len(audio_tensor), end_idx)
            
            # Extract segment if valid
            if end_idx > start_idx:
                segment = audio_tensor[start_idx:end_idx]
                segments.append(segment)
            else:
                logger.warning(f"Invalid segment: start={vad_seg.start}, end={vad_seg.end}")
        
        return segments

    async def __call__(self, 
                       audio_bytes: bytes, 
                       timestamp_granularity: Optional[Literal["word", "segment"]] = None):
        """
        Process audio through VAD+ASR pipeline.

        Args:
            audio_bytes: Audio data as bytes
            timestamp_granularity: Timestamp requirement (word/segment/None)

        Returns:
            Transcription response in appropriate format
        """
        # Load audio as tensor for VAD processing
        audio_tensor = await load_audio_from_bytes(audio_bytes)

        # Run VAD detection to find speech segments
        vad_segments: List[VADSegment] = await self.vad.remote(audio_bytes)
        logger.info(f"VAD detected {len(vad_segments)} segments")

        if not vad_segments:
            logger.warning("No speech segments detected by VAD")
            # Return empty transcription
            output_type = get_transcription_type(timestamp_granularity)
            if output_type == TranscriptionType.Text:
                return TranscriptionResponse(text="", usage=Usage(input_tokens=0, input_token_details=InputTokenDetails(text_tokens=0, audio_tokens=0), output_tokens=0, total_tokens=0))
            elif output_type == TranscriptionType.Word:
                return WordResponse(text="", language="unknown", duration=0.0, usage=DurationUsage(seconds=0), words=[])
            elif output_type == TranscriptionType.Segment:
                return SegmentResponse(text="", language="unknown", duration=0.0, usage=DurationUsage(seconds=0), segments=[])

        # Extract speech segments from audio
        speech_segments = self._extract_audio_segments(audio_tensor, vad_segments)

        # Send each segment separately so Serve can batch them across requests
        timestamp_granularities = [timestamp_granularity] * len(speech_segments)
        tasks = [
            self.asr.batched_transcribe_tensors.remote(seg, tg)
            for seg, tg in zip(speech_segments, timestamp_granularities)
        ]
        logger.info(f"_call - Starting batched ASR transcription for {len(tasks)} segments")
        transcriptions = await asyncio.gather(*tasks)

        # Combine transcriptions from all segments
        combined_text = " ".join([t.text for t in transcriptions])
        
        # Determine response format
        output_type = get_transcription_type(timestamp_granularity)
        duration = round(estimate_audio_duration(audio_bytes), 3)

        if output_type == TranscriptionType.Text:
            # Simple text transcription with token usage
            output_tokens = await asyncio.to_thread(approximate_count_tokens, combined_text)
            return TranscriptionResponse(
                text=combined_text,
                usage=Usage(
                    input_tokens=0,
                    input_token_details=InputTokenDetails(
                        text_tokens=0,
                        audio_tokens=0
                    ),
                    output_tokens=output_tokens,
                    total_tokens=output_tokens
                )
            )

        elif output_type == TranscriptionType.Word:
            # Word-level transcription with timestamps
            # Adjust word timestamps based on VAD segment offsets
            all_words = []
            for i, (transcription, vad_seg) in enumerate(zip(transcriptions, vad_segments)):
                for word in transcription.words:
                    # Adjust word timestamp by VAD segment start time
                    adjusted_word = word.model_copy(update={
                        'start': word.start + vad_seg.start,
                        'end': word.end + vad_seg.start
                    })
                    all_words.append(adjusted_word)
            
            lang_property = LanguageDetector.detect(combined_text, True)
            return WordResponse(
                text=combined_text,
                language=lang_property.language,
                duration=duration,
                usage=DurationUsage(seconds=math.ceil(duration)),
                words=[word.model_dump() for word in all_words]
            )

        elif output_type == TranscriptionType.Segment:
            # Segment-level transcription with timestamps
            # Adjust segment timestamps based on VAD segment offsets
            all_segments = []
            for i, (transcription, vad_seg) in enumerate(zip(transcriptions, vad_segments)):
                for segment in transcription.segments:
                    adjusted_segment = AdvancedTranscribedSegment(
                        id=len(all_segments),
                        start=segment.start + vad_seg.start,
                        end=segment.end + vad_seg.start,
                        text=segment.text
                    )
                    all_segments.append(adjusted_segment)
            
            lang_property = LanguageDetector.detect(combined_text, True)
            return SegmentResponse(
                text=combined_text,
                language=lang_property.language,
                duration=duration,
                usage=DurationUsage(seconds=math.ceil(duration)),
                segments=all_segments
            )


@serve.deployment(ray_actor_options={"num_gpus": INGRESS_NUM_GPUS},
                  num_replicas=INGRESS_NUM_REPLICAS,
                  max_ongoing_requests=INGRESS_MAX_ONGOING_REQUESTS)
@serve.ingress(ingress_app)
class IngressDeployment:
    """
    Ingress deployment that provides HTTP endpoints for the VAD+ASR pipeline.

    This deployment holds the FastAPI endpoints and delegates processing
    to the Pipeline deployment.
    """

    def __init__(self, pipeline: DeploymentHandle):
        """
        Initialize the ingress deployment with pipeline handle.

        Args:
            pipeline: Pipeline deployment handle
        """
        self.pipeline = pipeline

    @ingress_app.post("/v1/audio/transcriptions",
                      name="Transcribe audio files with optional timestamps",
                      tags=["Audio"])
    async def transcribe_audio(self,
                               file: UploadFile = File(...),
                               model: str = Form(ASR_MODEL_NAME),
                               timestamp_granularity: Optional[Literal["word", "segment"]] = Form(
                                   default=None,
                                   alias="timestamp_granularities[]",
                                   description="Level of timestamp detail: 'word' for word-level timestamps, 'segment' for segment-level timestamps, or None for no timestamps"),
                               response_format: str = Form("verbose_json")):
        """
        ## Transcribe audio files with optional timestamps using VAD+ASR pipeline.

        ### Args:
        - `file`: Audio file to transcribe
        - `model`: ASR model name (must match configured model)
        - `timestamp_granularity`: Level of timestamp detail (word/segment)
        - `response_format`: Output format (currently verbose_json)

        ### Returns:
        - Transcription response in requested format

        ### Raises:
        - `TranscriptedModelNotFoundException`: If requested model is not available
        - `UnsupportedAudioFormatException`: If audio format is not supported
        """
        logger.info(f"ASR request received - model: {model}")

        # Read uploaded audio file into memory for validation and processing
        audio_bytes = await file.read()

        # Validate that the uploaded file is a valid audio format
        if not is_audio_file(audio_bytes):
            # Extract file extension from filename without the dot (e.g., "mp3", "wav")
            file_extension = Path(file.filename).suffix.lstrip('.') if file.filename else ""
            raise UnsupportedAudioFormatException(file_format=file_extension)

        # Process audio through pipeline
        result = await self.pipeline.remote(audio_bytes, timestamp_granularity)
        
        return result


# Application binding function
def bind_app():
    """
    Create and bind the deployment graph.
    
    Returns:
        Bound ingress deployment
    """
    from app.services.deployments.vad_deployment import VADDeployment
    from app.services.deployments.asr_deployment import ASRDeployment
    
    # Create deployment instances
    vad = VADDeployment.bind()
    asr = ASRDeployment.bind()
    pipeline = Pipeline.bind(vad, asr)
    ingress = IngressDeployment.bind(pipeline)
    
    return ingress
