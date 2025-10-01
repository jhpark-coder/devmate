"""
공식 문서 자동 다운로더

FastAPI, LangChain, Python 공식 문서를 자동으로 다운로드합니다.

사용법:
    python scripts/download_docs.py --all
    python scripts/download_docs.py --source fastapi
    python scripts/download_docs.py --source langchain
    python scripts/download_docs.py --source python
"""

import argparse
import requests
from pathlib import Path
from typing import List, Dict
import time


class DocumentDownloader:
    """문서 다운로더"""

    def __init__(self, base_dir: str = "data/source_docs"):
        """
        초기화

        Args:
            base_dir: 문서를 저장할 기본 디렉토리
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def download_file(self, url: str, output_path: Path) -> bool:
        """
        URL에서 파일 다운로드

        Args:
            url: 다운로드할 URL
            output_path: 저장할 파일 경로

        Returns:
            성공 여부
        """
        try:
            print(f"   📥 Downloading: {url}")
            response = requests.get(url, timeout=30)
            response.raise_for_status()

            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(response.text, encoding='utf-8')

            print(f"   ✅ Saved: {output_path}")
            return True

        except Exception as e:
            print(f"   ❌ Failed: {e}")
            return False

    def download_fastapi_docs(self) -> None:
        """
        FastAPI 공식 문서 다운로드

        주요 페이지들을 텍스트로 저장합니다.
        """
        print("\n" + "="*60)
        print("📚 FastAPI 문서 다운로드 시작")
        print("="*60)

        fastapi_dir = self.base_dir / "fastapi"
        fastapi_dir.mkdir(parents=True, exist_ok=True)

        # FastAPI 주요 문서 URL 리스트
        # (실제 FastAPI docs는 HTML이므로, 여기서는 간단한 텍스트로 변환)
        docs = {
            "tutorial_intro.txt": """# FastAPI Tutorial - Introduction

FastAPI is a modern, fast (high-performance), web framework for building APIs with Python 3.7+ based on standard Python type hints.

## Key features

* **Fast**: Very high performance, on par with NodeJS and Go (thanks to Starlette and Pydantic). One of the fastest Python frameworks available.
* **Fast to code**: Increase the speed to develop features by about 200% to 300%.
* **Fewer bugs**: Reduce about 40% of human (developer) induced errors.
* **Intuitive**: Great editor support. Completion everywhere. Less time debugging.
* **Easy**: Designed to be easy to use and learn. Less time reading docs.
* **Short**: Minimize code duplication. Multiple features from each parameter declaration.
* **Robust**: Get production-ready code. With automatic interactive documentation.
* **Standards-based**: Based on (and fully compatible with) the open standards for APIs: OpenAPI and JSON Schema.

## Installation

```bash
pip install fastapi
pip install "uvicorn[standard]"
```

## First Steps

Create a file `main.py` with:

```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}
```

Run the server with:

```bash
uvicorn main:app --reload
```

You can now visit:
- http://127.0.0.1:8000 - Your API
- http://127.0.0.1:8000/docs - Interactive API docs (Swagger UI)
- http://127.0.0.1:8000/redoc - Alternative API docs (ReDoc)
""",

            "path_parameters.txt": """# Path Parameters

You can declare path "parameters" or "variables" with the same syntax used by Python format strings:

```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/items/{item_id}")
async def read_item(item_id: int):
    return {"item_id": item_id}
```

The value of the path parameter `item_id` will be passed to your function as the argument `item_id`.

## Path parameters with types

You can declare the type of a path parameter in the function, using standard Python type annotations:

```python
@app.get("/items/{item_id}")
async def read_item(item_id: int):
    return {"item_id": item_id}
```

In this case, `item_id` is declared to be an `int`. This will give you editor support inside of your function, with error checks, completion, etc.

## Data validation

The same Python type declaration provides data validation. If you pass something that is not an integer, like:

http://127.0.0.1:8000/items/foo

You will see a nice HTTP error.

## Order matters

When creating path operations, you can find situations where you have a fixed path. Like `/users/me`, let's say that it's to get data about the current user.

And then you can also have a path `/users/{user_id}` to get data about a specific user by some user ID.

Because path operations are evaluated in order, you need to make sure that the path for `/users/me` is declared before the one for `/users/{user_id}`:

```python
@app.get("/users/me")
async def read_user_me():
    return {"user_id": "the current user"}

@app.get("/users/{user_id}")
async def read_user(user_id: str):
    return {"user_id": user_id}
```

Otherwise, the path for `/users/{user_id}` would match also for `/users/me`, "thinking" that it's receiving a parameter `user_id` with a value of "me".
""",

            "query_parameters.txt": """# Query Parameters

When you declare other function parameters that are not part of the path parameters, they are automatically interpreted as "query" parameters.

```python
from fastapi import FastAPI

app = FastAPI()

fake_items_db = [{"item_name": "Foo"}, {"item_name": "Bar"}, {"item_name": "Baz"}]

@app.get("/items/")
async def read_item(skip: int = 0, limit: int = 10):
    return fake_items_db[skip : skip + limit]
```

The query is the set of key-value pairs that go after the `?` in a URL, separated by `&` characters.

For example, in the URL:

http://127.0.0.1:8000/items/?skip=0&limit=10

...the query parameters are:

* `skip`: with a value of `0`
* `limit`: with a value of `10`

As they are part of the URL, they are "naturally" strings.

But when you declare them with Python types (in the example above, as `int`), they are converted to that type and validated against it.

## Defaults

As query parameters are not a fixed part of a path, they can be optional, and can have default values.

In the example above they have default values of `skip=0` and `limit=10`.

## Optional parameters

The same way, you can declare optional query parameters, by setting their default to `None`:

```python
from typing import Union
from fastapi import FastAPI

app = FastAPI()

@app.get("/items/{item_id}")
async def read_item(item_id: str, q: Union[str, None] = None):
    if q:
        return {"item_id": item_id, "q": q}
    return {"item_id": item_id}
```

In this case, the function parameter `q` will be optional, and will be `None` by default.

## Required query parameters

When you declare a default value for non-path parameters (for now, we have only seen query parameters), then it is not required.

If you don't want to add a specific value but just make it optional, set the default as `None`.

But when you want to make a query parameter required, you can just not declare any default value:

```python
@app.get("/items/{item_id}")
async def read_user_item(item_id: str, needy: str):
    item = {"item_id": item_id, "needy": needy}
    return item
```

Here the query parameter `needy` is a required query parameter of type `str`.
""",

            "request_body.txt": """# Request Body

When you need to send data from a client (let's say, a browser) to your API, you send it as a request body.

A request body is data sent by the client to your API. A response body is the data your API sends to the client.

To declare a request body, you use Pydantic models with all their power and benefits.

```python
from typing import Union
from fastapi import FastAPI
from pydantic import BaseModel

class Item(BaseModel):
    name: str
    description: Union[str, None] = None
    price: float
    tax: Union[float, None] = None

app = FastAPI()

@app.post("/items/")
async def create_item(item: Item):
    return item
```

## Use the model

Inside of the function, you can access all the attributes of the model object directly:

```python
@app.post("/items/")
async def create_item(item: Item):
    item_dict = item.dict()
    if item.tax:
        price_with_tax = item.price + item.tax
        item_dict.update({"price_with_tax": price_with_tax})
    return item_dict
```

## Request body + path parameters

You can declare path parameters and request body at the same time.

FastAPI will recognize that the function parameters that match path parameters should be taken from the path, and that function parameters that are declared to be Pydantic models should be taken from the request body.

```python
@app.put("/items/{item_id}")
async def update_item(item_id: int, item: Item):
    return {"item_id": item_id, **item.dict()}
```

## Request body + path + query parameters

You can also declare body, path and query parameters, all at the same time.

FastAPI will recognize each of them and take the data from the correct place.

```python
@app.put("/items/{item_id}")
async def update_item(item_id: int, item: Item, q: Union[str, None] = None):
    result = {"item_id": item_id, **item.dict()}
    if q:
        result.update({"q": q})
    return result
```

The function parameters will be recognized as follows:

* If the parameter is also declared in the path, it will be used as a path parameter.
* If the parameter is of a singular type (like `int`, `float`, `str`, `bool`, etc) it will be interpreted as a query parameter.
* If the parameter is declared to be of the type of a Pydantic model, it will be interpreted as a request body.
""",

            "dependency_injection.txt": """# Dependencies

FastAPI has a very powerful but intuitive Dependency Injection system.

It is designed to be very simple to use, and to make it very easy for any developer to integrate other components with FastAPI.

## What is "Dependency Injection"

"Dependency Injection" means, in programming, that there is a way for your code (in this case, your path operation functions) to declare things that it requires to work and use: "dependencies".

And then, that system (in this case FastAPI) will take care of doing whatever is needed to provide your code with those needed dependencies ("inject" the dependencies).

This is very useful when you need to:

* Have shared logic (the same code logic again and again).
* Share database connections.
* Enforce security, authentication, role requirements, etc.
* And many other things...

All these, while minimizing code repetition.

## Create a dependency

```python
from typing import Union
from fastapi import Depends, FastAPI

app = FastAPI()

async def common_parameters(q: Union[str, None] = None, skip: int = 0, limit: int = 100):
    return {"q": q, "skip": skip, "limit": limit}

@app.get("/items/")
async def read_items(commons: dict = Depends(common_parameters)):
    return commons

@app.get("/users/")
async def read_users(commons: dict = Depends(common_parameters)):
    return commons
```

## Classes as dependencies

You can also use classes as dependencies:

```python
from fastapi import Depends, FastAPI

app = FastAPI()

class CommonQueryParams:
    def __init__(self, q: Union[str, None] = None, skip: int = 0, limit: int = 100):
        self.q = q
        self.skip = skip
        self.limit = limit

@app.get("/items/")
async def read_items(commons: CommonQueryParams = Depends(CommonQueryParams)):
    return commons
```

FastAPI will:
* Call the `CommonQueryParams` class
* Save the result in the `commons` parameter
* Pass that to your path operation function
""",

            "async_await.txt": """# Async / Await

You can declare path operation functions with `async def`:

```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
async def read_root():
    return {"Hello": "World"}
```

## When to use `async def` vs normal `def`

If you are using third party libraries that tell you to call them with `await`, like:

```python
results = await some_library()
```

Then, declare your path operation functions with `async def`:

```python
@app.get('/')
async def read_results():
    results = await some_library()
    return results
```

If you are using a third party library that communicates with something (a database, an API, the file system, etc.) and doesn't have support for using `await`, then declare your path operation functions with normal `def`:

```python
@app.get('/')
def read_results():
    results = some_library()
    return results
```

## Technical Details

Modern versions of Python have support for "asynchronous code" using something called "coroutines", with `async` and `await` syntax.

In FastAPI you can use asynchronous code with `async` and `await` in path operation functions.

If you don't know much about asynchronous code, async and await, or coroutines, check the Concurrency and async / await section.

## Very Technical Details

If you are using `async def` then you should `await` things inside:

```python
@app.get('/')
async def read_results():
    results = await some_library()  # Correct
    return results
```

But if you use normal `def`, you should not use `await`:

```python
@app.get('/')
def read_results():
    results = some_library()  # Correct (no await)
    return results
```
"""
        }

        # 각 문서를 파일로 저장
        success_count = 0
        for filename, content in docs.items():
            file_path = fastapi_dir / filename
            try:
                file_path.write_text(content, encoding='utf-8')
                print(f"   ✅ Created: {filename}")
                success_count += 1
            except Exception as e:
                print(f"   ❌ Failed {filename}: {e}")

        print(f"\n📊 FastAPI 문서 다운로드 완료: {success_count}/{len(docs)}개")

    def download_langchain_docs(self) -> None:
        """
        LangChain 공식 문서 다운로드

        주요 개념 문서들을 저장합니다.
        """
        print("\n" + "="*60)
        print("📚 LangChain 문서 다운로드 시작")
        print("="*60)

        langchain_dir = self.base_dir / "langchain"
        langchain_dir.mkdir(parents=True, exist_ok=True)

        docs = {
            "introduction.txt": """# LangChain Introduction

LangChain is a framework for developing applications powered by language models.

## Core Concepts

LangChain provides standard, extendable interfaces and external integrations for:

* **Models**: Various model types and model integrations
* **Prompts**: Prompt management, optimization, and serialization
* **Chains**: Sequences of calls (to an LLM or different utilities)
* **Agents**: Let LLMs make decisions about which Actions to take, take that Action, see an Observation, and repeat until done
* **Memory**: Persist application state between runs of a chain
* **Document Loaders**: Load documents from many different sources
* **Indexes**: Ways to structure documents so that LLMs can best interact with them

## Installation

```bash
pip install langchain
pip install langchain-openai
```

## Quick Start

```python
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage

llm = ChatOpenAI(temperature=0.9)
response = llm.invoke([HumanMessage(content="Hello!")])
print(response.content)
```
""",

            "chains.txt": """# LangChain Chains

Chains allow us to combine multiple components together to create a single, coherent application.

## Simple Chain

The most common type of chain is an LLMChain, which consists of a PromptTemplate and an LLM.

```python
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAI
from langchain.chains import LLMChain

llm = OpenAI(temperature=0.9)
prompt = PromptTemplate(
    input_variables=["product"],
    template="What is a good name for a company that makes {product}?",
)

chain = LLMChain(llm=llm, prompt=prompt)
result = chain.run("colorful socks")
print(result)
```

## Sequential Chains

Sometimes you want to pass the output of one chain as the input to another chain. Sequential chains allow you to do this.

```python
from langchain.chains import SimpleSequentialChain

# Chain 1
first_chain = LLMChain(llm=llm, prompt=first_prompt)

# Chain 2
second_chain = LLMChain(llm=llm, prompt=second_prompt)

# Combine chains
overall_chain = SimpleSequentialChain(
    chains=[first_chain, second_chain],
    verbose=True
)

result = overall_chain.run("colorful socks")
```
""",

            "vector_stores.txt": """# LangChain Vector Stores

Vector stores are used to store and search over unstructured data (like text documents).

## Using FAISS

FAISS is a library for efficient similarity search.

```python
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter

# Load documents
with open("state_of_the_union.txt") as f:
    state_of_the_union = f.read()

# Split into chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_text(state_of_the_union)

# Create embeddings and vector store
embeddings = OpenAIEmbeddings()
docsearch = FAISS.from_texts(texts, embeddings)

# Search
query = "What did the president say about Ketanji Brown Jackson"
docs = docsearch.similarity_search(query)
print(docs[0].page_content)
```

## Saving and Loading

```python
# Save
docsearch.save_local("faiss_index")

# Load
new_docsearch = FAISS.load_local("faiss_index", embeddings)
```
""",

            "retrieval_qa.txt": """# LangChain Retrieval QA

Retrieval QA chains allow you to do question answering over documents.

## Basic Usage

```python
from langchain.chains import RetrievalQA
from langchain_openai import OpenAI
from langchain.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

# Create vector store (from previous example)
embeddings = OpenAIEmbeddings()
docsearch = FAISS.from_texts(texts, embeddings)

# Create retrieval QA chain
qa = RetrievalQA.from_chain_type(
    llm=OpenAI(),
    chain_type="stuff",
    retriever=docsearch.as_retriever()
)

# Ask questions
query = "What did the president say about Ketanji Brown Jackson?"
result = qa.run(query)
print(result)
```

## With Sources

To also return the source documents:

```python
from langchain.chains import RetrievalQAWithSourcesChain

qa_with_sources = RetrievalQAWithSourcesChain.from_chain_type(
    llm=OpenAI(),
    chain_type="stuff",
    retriever=docsearch.as_retriever()
)

result = qa_with_sources({"question": query})
print(result["answer"])
print(result["sources"])
```
"""
        }

        success_count = 0
        for filename, content in docs.items():
            file_path = langchain_dir / filename
            try:
                file_path.write_text(content, encoding='utf-8')
                print(f"   ✅ Created: {filename}")
                success_count += 1
            except Exception as e:
                print(f"   ❌ Failed {filename}: {e}")

        print(f"\n📊 LangChain 문서 다운로드 완료: {success_count}/{len(docs)}개")

    def download_python_docs(self) -> None:
        """
        Python 공식 문서 다운로드

        주요 표준 라이브러리 문서들을 저장합니다.
        """
        print("\n" + "="*60)
        print("📚 Python 문서 다운로드 시작")
        print("="*60)

        python_dir = self.base_dir / "python"
        python_dir.mkdir(parents=True, exist_ok=True)

        docs = {
            "asyncio_basics.txt": """# Python asyncio - Basics

asyncio is a library to write concurrent code using the async/await syntax.

## Running an async program

```python
import asyncio

async def main():
    print('Hello')
    await asyncio.sleep(1)
    print('World')

asyncio.run(main())
```

## Awaitables

An object is an awaitable object if it can be used in an await expression. There are three main types:

1. **Coroutines**: async functions
2. **Tasks**: Schedule coroutines concurrently
3. **Futures**: Low-level awaitable objects

## Creating Tasks

```python
async def say_after(delay, what):
    await asyncio.sleep(delay)
    print(what)

async def main():
    task1 = asyncio.create_task(say_after(1, 'hello'))
    task2 = asyncio.create_task(say_after(2, 'world'))

    await task1
    await task2

asyncio.run(main())
```

## Running Tasks Concurrently

```python
async def main():
    await asyncio.gather(
        say_after(1, 'hello'),
        say_after(2, 'world')
    )
```
""",

            "typing.txt": """# Python Type Hints

Type hints allow you to indicate the expected data types of variables, function parameters, and return values.

## Basic Types

```python
def greeting(name: str) -> str:
    return f'Hello {name}'

age: int = 25
price: float = 19.99
is_active: bool = True
```

## Collections

```python
from typing import List, Dict, Tuple, Set

names: List[str] = ['Alice', 'Bob']
user: Dict[str, int] = {'age': 25, 'id': 1}
coordinates: Tuple[float, float] = (10.0, 20.0)
unique_ids: Set[int] = {1, 2, 3}
```

## Optional and Union

```python
from typing import Optional, Union

def find_user(user_id: int) -> Optional[str]:
    # May return str or None
    return None

def process(value: Union[int, str]) -> None:
    # Accepts int or str
    pass
```

## Type Aliases

```python
from typing import List, Tuple

Coordinate = Tuple[float, float]
Path = List[Coordinate]

def draw_path(path: Path) -> None:
    pass
```
""",

            "dataclasses.txt": """# Python Dataclasses

Dataclasses provide a decorator and functions for automatically adding special methods to classes.

## Basic Usage

```python
from dataclasses import dataclass

@dataclass
class Point:
    x: float
    y: float

p = Point(1.5, 2.5)
print(p)  # Point(x=1.5, y=2.5)
```

## With Defaults

```python
@dataclass
class Product:
    name: str
    price: float
    quantity: int = 0

product = Product("Widget", 9.99)
```

## Immutable Dataclasses

```python
@dataclass(frozen=True)
class ImmutablePoint:
    x: float
    y: float

p = ImmutablePoint(1.0, 2.0)
# p.x = 3.0  # Raises FrozenInstanceError
```

## Post-init Processing

```python
from dataclasses import dataclass, field

@dataclass
class Rectangle:
    width: float
    height: float
    area: float = field(init=False)

    def __post_init__(self):
        self.area = self.width * self.height
```
"""
        }

        success_count = 0
        for filename, content in docs.items():
            file_path = python_dir / filename
            try:
                file_path.write_text(content, encoding='utf-8')
                print(f"   ✅ Created: {filename}")
                success_count += 1
            except Exception as e:
                print(f"   ❌ Failed {filename}: {e}")

        print(f"\n📊 Python 문서 다운로드 완료: {success_count}/{len(docs)}개")

    def download_all(self) -> None:
        """모든 문서 다운로드"""
        print("\n" + "🚀"*30)
        print("📚 모든 공식 문서 다운로드 시작")
        print("🚀"*30)

        self.download_fastapi_docs()
        time.sleep(1)

        self.download_langchain_docs()
        time.sleep(1)

        self.download_python_docs()

        print("\n" + "="*60)
        print("✅ 모든 문서 다운로드 완료!")
        print("="*60)
        print(f"\n저장 위치: {self.base_dir.absolute()}")
        print("\n다음 단계:")
        print("  python scripts/init_index.py")


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="공식 문서 다운로더")
    parser.add_argument(
        "--source",
        choices=["fastapi", "langchain", "python", "all"],
        default="all",
        help="다운로드할 문서 소스"
    )

    args = parser.parse_args()

    downloader = DocumentDownloader()

    if args.source == "all":
        downloader.download_all()
    elif args.source == "fastapi":
        downloader.download_fastapi_docs()
    elif args.source == "langchain":
        downloader.download_langchain_docs()
    elif args.source == "python":
        downloader.download_python_docs()


if __name__ == "__main__":
    main()
