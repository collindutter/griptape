from attr import define

from griptape.utils import PromptStack
from griptape.drivers import BasePromptDriver
from griptape.tokenizers import TiktokenTokenizer, BaseTokenizer
from griptape.artifacts import TextArtifact


@define
class MockFailingPromptDriver(BasePromptDriver):
    max_failures: int
    current_attempt: int = 0
    model: str = "test-model"
    tokenizer: BaseTokenizer = TiktokenTokenizer()

    def try_run(self, prompt_stack: PromptStack) -> TextArtifact:
        if self.current_attempt < self.max_failures:
            self.current_attempt += 1

            raise Exception("failed attempt")
        else:
            return TextArtifact("success")
