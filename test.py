from camel.agents import ChatAgent
from camel.models import ModelFactory
from camel.types import ModelPlatformType

model = ModelFactory.create(
    model_platform=ModelPlatformType.OPENAI_COMPATIBLE_MODEL,
    model_type="qwen3-max",
    url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key="sk-e1d4deaca8734f4fbfed0c77dfc6ca23",
)

agent = ChatAgent(
    model=model,
    output_language="中文",
)

response = agent.step("你好，你是谁?")
print(response.msgs[0].content)
