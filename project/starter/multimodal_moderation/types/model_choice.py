from pydantic import BaseModel, ConfigDict
from pydantic_ai.models import Model
from pydantic_ai.settings import ModelSettings


class ModelChoice(BaseModel):
    model: Model | str
    model_settings: ModelSettings | None

    # Allow arbitrary types so we can store the Model directly.
    model_config = ConfigDict(arbitrary_types_allowed=True)
