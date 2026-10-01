from pydantic import BaseModel
from datasets import Dataset, DatasetDict, load_dataset
from typing import Optional, Self


PREFIX = "Price is $"
QUESTION = "What does this cost to the nearest dollar?"


class Item(BaseModel):
    """
    An Item is a data-point of a Product with a Price
    """

    title: str
    category: str
    price: float
    full: Optional[str] = None
    weight: Optional[float] = None
    summary: Optional[str] = None
    prompt: Optional[str] = None
    id: Optional[int] = None

    # Builds a training prompt for the item and stores it in self.prompt. 
    # It doesn't return anything; it modifies the item in place. An example for the prompt generated:
    #
    # What does this cost to the nearest dollar?
    #
    # Stainless steel water bottle, 32oz
    #
    # Price is $25.00
    def make_prompt(self, text: str):
        self.prompt = f"{QUESTION}\n\n{text}\n\n{PREFIX}{round(self.price)}.00"

    # The counterpart to make_prompt(). It takes that same string and splits on PREFIX, keeping only the 
    # part before it, then adds PREFIX back. That gives you everything up to Price is $ with the answer cut off, 
    # which is what you'd feed a model at inference time so it has to fill in the price itself:
    #
    # What does this cost to the nearest dollar?
    #
    # Stainless steel water bottle, 32oz
    #
    # Price is $
    def test_prompt(self) -> str:
        return self.prompt.split(PREFIX)[0] + PREFIX

    # Defines how an Item is displayed when Python needs a string representation of it, such as when you print 
    # it in a list, look at it in a notebook cell, or inspect it in a debugger.
    # It returns a compact one-liner with the title and price:
    #
    # <Stainless steel water bottle = $24.6>
    def __repr__(self) -> str:
        return f"<{self.title} = ${self.price}>"

    @staticmethod
    def push_to_hub(dataset_name: str, train: list[Self], val: list[Self], test: list[Self]):
        """Push Item lists to HuggingFace Hub"""
        DatasetDict(
            {
                "train": Dataset.from_list([item.model_dump() for item in train]),
                "validation": Dataset.from_list([item.model_dump() for item in val]),
                "test": Dataset.from_list([item.model_dump() for item in test]),
            }
        ).push_to_hub(dataset_name)

    @classmethod
    def from_hub(cls, dataset_name: str) -> tuple[list[Self], list[Self], list[Self]]:
        """Load from HuggingFace Hub and reconstruct Items"""
        ds = load_dataset(dataset_name)
        return (
            # Build an object from a dict using Pydantic's model_validate().
            # It checks for any mismatched dict fields against the Item schema 
            # (title, category, price, and the optional full, weight, summary, prompt, id), 
            # and raises an error if a row doesn't fit.
            [cls.model_validate(row) for row in ds["train"]],
            [cls.model_validate(row) for row in ds["validation"]],
            [cls.model_validate(row) for row in ds["test"]],
        )
