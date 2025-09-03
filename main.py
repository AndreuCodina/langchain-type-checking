import asyncio
from typing import TYPE_CHECKING, Any

from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, SecretStr

if TYPE_CHECKING:
    from langchain_core.runnables import Runnable


class GetFundingInput(BaseModel):
    company: str


class GetFundingOutput(BaseModel):
    total_funding: str


async def get_funding(company: str) -> None:
    input_ = GetFundingInput(company=company)
    prompt = ChatPromptTemplate(
        [
            SystemMessagePromptTemplate.from_template(
                """
Total funding received by the provided company.

<company>
{company}
</company>
""".strip()
            ),
        ],
        input_variables=list(GetFundingInput.model_fields),
    )
    model: Runnable[dict[str, Any], GetFundingOutput] = (  # type: ignore[reportUnknownVariableType]
        ChatOpenAI(  # type: ignore[reportUnknownVariableType]
            api_key=SecretStr("TBD"),
            model="gpt-5-mini-2025-08-07",
            temperature=0.0,
        ).with_structured_output(GetFundingOutput, strict=True)
    )
    chain: Runnable[dict[str, Any], GetFundingOutput] = (  # type: ignore[reportUnknownVariableType]
        prompt | model
    ).with_config({"run_name": get_funding.__name__})
    output = await chain.ainvoke(input_.model_dump())
    print(f"Total funding for {company}: {output.total_funding}")  # noqa: T201


if __name__ == "__main__":
    asyncio.run(get_funding(company="Tesla"))
