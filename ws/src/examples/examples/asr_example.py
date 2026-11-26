#!/usr/bin/env python3

import wave

import numpy as np
import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import ParameterDescriptor
from audio_utils import pcm2float
from pepper_interfaces.srv import Asr

DEFAULT_WAV = "/workspace/ws/src/pepper_nodes/pepper_nodes/audio.wav"
TARGET_SR = 16000  # Whisper expects 16 kHz mono audio


def load_wav(path: str, target_sr: int = TARGET_SR) -> np.ndarray:
    """Read a PCM wav file and return a mono float32 array in [-1, 1] at target_sr."""
    with wave.open(path, "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        sr = wf.getframerate()
        frames = wf.readframes(wf.getnframes())

    dtypes = {1: np.uint8, 2: np.int16, 4: np.int32}
    if sampwidth not in dtypes:
        raise ValueError(f"Unsupported sample width: {sampwidth * 8} bit")
    audio = pcm2float(np.frombuffer(frames, dtype=dtypes[sampwidth]))

    if n_channels > 1:
        audio = audio.reshape(-1, n_channels).mean(axis=1)

    if sr != target_sr:
        duration = len(audio) / sr
        n_out = int(round(duration * target_sr))
        audio = np.interp(
            np.linspace(0.0, duration, n_out, endpoint=False),
            np.arange(len(audio)) / sr,
            audio,
        )

    return audio.astype(np.float32)


class ASRClient(Node):

    def __init__(self):
        super().__init__("asr_node_example")
        self.declare_parameter(
            "wav_path", DEFAULT_WAV,
            descriptor=ParameterDescriptor(description="Path of the wav file to transcribe"))
        self.wav_path = self.get_parameter("wav_path").get_parameter_value().string_value

        self.cli = self.create_client(Asr, "/asr")
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Waiting for /asr service...")

    def call(self, audio: np.ndarray):
        request = Asr.Request()
        request.audio_features = audio.tolist()

        future = self.cli.call_async(request)
        rclpy.spin_until_future_complete(self, future)

        if future.result() is not None:
            self.get_logger().info(f"Transcription: {future.result().text}")
            return future.result().text
        self.get_logger().error("Service call failed")
        return None


def main():
    rclpy.init()
    node = ASRClient()

    audio = load_wav(node.wav_path)
    node.get_logger().info(
        f"Loaded {node.wav_path} ({len(audio) / TARGET_SR:.2f} s), sending to /asr...")
    node.call(audio)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
