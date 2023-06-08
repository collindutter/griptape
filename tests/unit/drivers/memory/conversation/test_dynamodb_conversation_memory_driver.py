import os
import pytest
from tests.mocks.mock_prompt_driver import MockPromptDriver
from griptape.memory.structure import ConversationMemory
from griptape.tasks import PromptTask
from griptape.structures import Pipeline
from griptape.drivers import DynamoDbConversationMemoryDriver
from moto import mock_dynamodb
from boto3 import resource, client


class TestDynamoDbConversationMemoryDriver:
    DYNAMODB_TABLE_NAME = "griptape"
    DYNAMODB_PARTITION_KEY = "entryId"
    AWS_REGION = "us-west-2"
    VALUE_ATTRIBUTE_KEY = "foo"
    PARTITION_KEY_VALUE = "bar"

    @pytest.fixture(autouse=True)
    def table_gen(self):
        self._mock_aws_credentials()
        self.mock_dynamodb = mock_dynamodb()
        self.mock_dynamodb.start()

        dynamodb = client("dynamodb", region_name=self.AWS_REGION)
        dynamodb.create_table(
            TableName=self.DYNAMODB_TABLE_NAME,
            KeySchema=[
                {"AttributeName": self.DYNAMODB_PARTITION_KEY, "KeyType": "HASH"}
            ],
            AttributeDefinitions=[
                {"AttributeName": self.DYNAMODB_PARTITION_KEY, "AttributeType": "S"},
            ],
            BillingMode="PAY_PER_REQUEST",
        )

        yield

        dynamodb.delete_table(TableName=self.DYNAMODB_TABLE_NAME)
        self.mock_dynamodb.stop()

    def test_store(self):
        dynamodb = resource("dynamodb", region_name=self.AWS_REGION)
        table = dynamodb.Table(self.DYNAMODB_TABLE_NAME)
        prompt_driver = MockPromptDriver()
        memory_driver = DynamoDbConversationMemoryDriver(
            aws_region=self.AWS_REGION,
            table_name=self.DYNAMODB_TABLE_NAME,
            partition_key=self.DYNAMODB_PARTITION_KEY,
            value_attribute_key=self.VALUE_ATTRIBUTE_KEY,
            partition_key_value=self.PARTITION_KEY_VALUE,
        )
        memory = ConversationMemory(driver=memory_driver)
        pipeline = Pipeline(prompt_driver=prompt_driver, memory=memory)

        pipeline.add_task(PromptTask("test"))

        response = table.get_item(
            TableName=self.DYNAMODB_TABLE_NAME, Key={"entryId": "bar"}
        )
        assert "Item" not in response

        pipeline.run()

        response = table.get_item(
            TableName=self.DYNAMODB_TABLE_NAME, Key={"entryId": "bar"}
        )
        assert "Item" in response

    def test_load(self):
        prompt_driver = MockPromptDriver()
        memory_driver = DynamoDbConversationMemoryDriver(
            aws_region=self.AWS_REGION,
            table_name=self.DYNAMODB_TABLE_NAME,
            partition_key=self.DYNAMODB_PARTITION_KEY,
            value_attribute_key=self.VALUE_ATTRIBUTE_KEY,
            partition_key_value=self.PARTITION_KEY_VALUE,
        )
        memory = ConversationMemory(driver=memory_driver)
        pipeline = Pipeline(prompt_driver=prompt_driver, memory=memory)

        pipeline.add_task(PromptTask("test"))

        pipeline.run()
        pipeline.run()

        new_memory = memory_driver.load()

        assert new_memory.type == "ConversationMemory"
        assert len(new_memory.runs) == 2
        assert new_memory.runs[0].input == "test"
        assert new_memory.runs[0].output == "mock output"

    def _mock_aws_credentials(self):
        os.environ["AWS_ACCESS_KEY_ID"] = "testing"
        os.environ["AWS_SECRET_ACCESS_KEY"] = "testing"
        os.environ["AWS_SECURITY_TOKEN"] = "testing"
        os.environ["AWS_SESSION_TOKEN"] = "testing"