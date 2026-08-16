"""HTTP routers. Thin layer: parse/validate, call domain modules, serialize.

No DSP or model logic lives here — routers call into audio/, alignment/,
scoring/, prosody/, measurements/, feedback/ and compose an AnalysisResult.
"""
