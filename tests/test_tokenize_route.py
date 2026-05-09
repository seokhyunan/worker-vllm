import asyncio
import importlib.util
import sys
import types
from pathlib import Path


_MISSING = object()
_ORIGINAL_MODULES = {}
_INSTALLED_MODULES = []


def _install_module(name, **attrs):
    if name not in _ORIGINAL_MODULES:
        _ORIGINAL_MODULES[name] = sys.modules.get(name, _MISSING)
    _INSTALLED_MODULES.append(name)

    module = types.ModuleType(name)
    module.__dict__.update(attrs)
    if "." not in name:
        module.__path__ = []
    sys.modules[name] = module

    if "." in name:
        parent_name, child_name = name.rsplit(".", 1)
        parent = sys.modules.get(parent_name)
        if parent is not None:
            setattr(parent, child_name, module)

    return module


def _restore_modules():
    for name in reversed(_INSTALLED_MODULES):
        original = _ORIGINAL_MODULES[name]
        if original is _MISSING:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = original


def _load_engine_module():
    src_dir = Path(__file__).resolve().parents[1] / "src"
    sys.path.insert(0, str(src_dir))
    try:
        spec = importlib.util.spec_from_file_location(
            "_worker_vllm_engine_under_test",
            src_dir / "engine.py",
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(src_dir))


class _DumpModel:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def model_dump(self, *args, **kwargs):
        return dict(self.__dict__)


class _TokenizeCompletionRequest(_DumpModel):
    def __init__(self, **kwargs):
        if "prompt" not in kwargs:
            raise ValueError("prompt is required")
        super().__init__(**kwargs)


class _TokenizeChatRequest(_DumpModel):
    def __init__(self, **kwargs):
        if "messages" not in kwargs:
            raise ValueError("messages is required")
        super().__init__(**kwargs)


class _TokenizeResponse(_DumpModel):
    pass


class _ErrorResponse:
    def __init__(self, message, code=400):
        self.error = types.SimpleNamespace(code=code)
        self.message = message

    def model_dump(self, *args, **kwargs):
        return {"error": {"message": self.message, "code": self.error.code}}


class _FakeTokenizationEngine:
    def __init__(self, response=None):
        self.response = response or _TokenizeResponse(
            tokens=[9906, 1917],
            token_strs=["Hello", " world"],
            count=2,
            max_model_len=4096,
        )
        self.request = None

    async def create_tokenize(self, request, raw_request):
        self.request = request
        return self.response


def _install_engine_stubs():
    _install_module("dotenv", load_dotenv=lambda: None)
    _install_module("vllm", AsyncLLMEngine=object)
    _install_module("vllm.entrypoints")
    _install_module("vllm.entrypoints.logger", RequestLogger=object)
    _install_module("vllm.entrypoints.anthropic")
    _install_module(
        "vllm.entrypoints.anthropic.protocol",
        AnthropicMessagesRequest=object,
        AnthropicMessagesResponse=object,
        AnthropicError=object,
        AnthropicErrorResponse=object,
    )
    _install_module("vllm.entrypoints.anthropic.serving", AnthropicServingMessages=object)
    _install_module("vllm.entrypoints.openai")
    _install_module("vllm.entrypoints.openai.chat_completion")
    _install_module(
        "vllm.entrypoints.openai.chat_completion.protocol",
        ChatCompletionRequest=object,
    )
    _install_module(
        "vllm.entrypoints.openai.chat_completion.serving",
        OpenAIServingChat=object,
    )
    _install_module("vllm.entrypoints.openai.completion")
    _install_module(
        "vllm.entrypoints.openai.completion.protocol",
        CompletionRequest=object,
    )
    _install_module(
        "vllm.entrypoints.openai.completion.serving",
        OpenAIServingCompletion=object,
    )
    _install_module("vllm.entrypoints.openai.engine")
    _install_module(
        "vllm.entrypoints.openai.engine.protocol",
        ErrorResponse=_ErrorResponse,
    )
    _install_module("vllm.entrypoints.openai.models")
    _install_module(
        "vllm.entrypoints.openai.models.protocol",
        BaseModelPath=object,
        LoRAModulePath=object,
    )
    _install_module(
        "vllm.entrypoints.openai.models.serving",
        OpenAIServingModels=object,
    )
    _install_module("vllm.entrypoints.openai.responses")
    _install_module(
        "vllm.entrypoints.openai.responses.protocol",
        ResponsesRequest=object,
        ResponsesResponse=object,
    )
    _install_module(
        "vllm.entrypoints.openai.responses.serving",
        OpenAIServingResponses=object,
    )
    _install_module("vllm.entrypoints.serve")
    _install_module("vllm.entrypoints.serve.render")
    _install_module(
        "vllm.entrypoints.serve.render.serving",
        OpenAIServingRender=object,
    )
    _install_module("vllm.entrypoints.serve.tokenize")
    _install_module(
        "vllm.entrypoints.serve.tokenize.protocol",
        TokenizeChatRequest=_TokenizeChatRequest,
        TokenizeCompletionRequest=_TokenizeCompletionRequest,
        TokenizeResponse=_TokenizeResponse,
    )
    _install_module(
        "vllm.entrypoints.serve.tokenize.serving",
        OpenAIServingTokenization=object,
    )
    _install_module("engine_args", get_engine_args=lambda: None)
    _install_module("tokenizer", TokenizerWrapper=object)
    _install_module(
        "utils",
        BatchSize=object,
        DummyRequest=object,
        JobInput=object,
        create_error_response=lambda message, **kwargs: _ErrorResponse(message),
    )


_install_engine_stubs()
try:
    engine = _load_engine_module()
finally:
    _restore_modules()


class _OpenAIRequest:
    def __init__(self, openai_input, openai_route="/v1/tokenize"):
        self.openai_route = openai_route
        self.openai_input = openai_input
        self.request_id = "test-request"


async def _noop():
    return None


async def _collect(async_generator):
    return [item async for item in async_generator]


def _make_engine(response=None):
    instance = engine.OpenAIvLLMEngine.__new__(engine.OpenAIvLLMEngine)
    instance.tokenization_engine = _FakeTokenizationEngine(response=response)
    instance._ensure_engines_initialized = _noop
    return instance


def test_generate_routes_v1_tokenize_to_tokenization_handler():
    instance = _make_engine()

    results = asyncio.run(_collect(instance.generate(
        _OpenAIRequest({"prompt": "Hello world"}, openai_route="/v1/tokenize")
    )))

    assert len(results) == 1
    assert results[0]["tokens"] == [9906, 1917]


def test_generate_routes_native_tokenize_to_tokenization_handler():
    instance = _make_engine()

    results = asyncio.run(_collect(instance.generate(
        _OpenAIRequest({"prompt": "Hello world"}, openai_route="/tokenize")
    )))

    assert len(results) == 1
    assert results[0]["count"] == 2


def test_tokenize_prompt_uses_completion_request():
    instance = _make_engine()

    result = asyncio.run(instance._handle_tokenize_request(
        _OpenAIRequest({"prompt": "Hello world", "return_token_strs": True})
    ))

    assert result["tokens"] == [9906, 1917]
    assert result["token_strs"] == ["Hello", " world"]
    assert isinstance(instance.tokenization_engine.request, _TokenizeCompletionRequest)


def test_tokenize_messages_uses_chat_request():
    instance = _make_engine()

    result = asyncio.run(instance._handle_tokenize_request(
        _OpenAIRequest({"messages": [{"role": "user", "content": "Hello!"}]})
    ))

    assert result["count"] == 2
    assert isinstance(instance.tokenization_engine.request, _TokenizeChatRequest)


def test_tokenize_prefers_prompt_when_prompt_and_messages_are_present():
    instance = _make_engine()

    asyncio.run(instance._handle_tokenize_request(
        _OpenAIRequest({
            "prompt": "Hello world",
            "messages": [{"role": "user", "content": "Hello!"}],
        })
    ))

    assert isinstance(instance.tokenization_engine.request, _TokenizeCompletionRequest)


def test_tokenize_missing_prompt_and_messages_returns_error_payload():
    instance = _make_engine()

    result = asyncio.run(instance._handle_tokenize_request(_OpenAIRequest({})))

    assert result == {
        "error": {
            "message": "Expected `prompt` or `messages`.",
            "code": 400,
        }
    }


def test_tokenize_returns_vllm_error_response_payload():
    error_response = _ErrorResponse("The model `missing` does not exist.", code=404)
    instance = _make_engine(response=error_response)

    result = asyncio.run(instance._handle_tokenize_request(
        _OpenAIRequest({"model": "missing", "prompt": "Hello world"})
    ))

    assert result == {
        "error": {
            "message": "The model `missing` does not exist.",
            "code": 404,
        }
    }
