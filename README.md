# iPDb


Experimental in-database LLM and deep-learning inference via native relational prediction operator. Built on top of a high-performance analytical database system, [DuckDB](https://github.com/duckdb/duckdb). 

Refer to the [Documentation](https://ipdb-docs.pages.dev) for usage.

Read the paper at [arxiv](https://arxiv.org/pdf/2601.16432).

```bibtex
@misc{ipdb2026arxiv,
      title={iPDB -- Optimizing Semantic SQL Queries}, 
      author={Udesh Kumarasinghe and Tyler Liu and Ahmed R. Mahmood and Chunwei Liu and Walid G. Aref},
      year={2026},
      eprint={2601.16432},
      archivePrefix={arXiv},
      primaryClass={cs.DB},
      url={https://arxiv.org/abs/2601.16432}, 
}
```

## Installing prebuilt binaries

We publish prebuilt Linux CLI binaries for `amd64` and `arm64`, built with LLM OpenAI API support enabled (i.e., `ENABLE_PREDICT=1 ENABLE_LLM_API=1`), but without the native ONNX/llama.cpp backends. These are available at [Releases](https://github.com/purduedb/iPDb/releases) — look for `ipdb-cli-linux-<arch>.zip` (or `.gz`) for the CLI shell.

```bash
curl -LO https://github.com/purduedb/iPDb/releases/download/v1.2.0/ipdb-cli-linux-amd64.zip
unzip ipdb-cli-linux-amd64.zip
chmod +x ipdb
./ipdb
```

> Replace architecture you want, `amd64` with `arm64` if you're on an ARM64 host.

## Installing the Python package

We release a `duckdb-python` package built with iPDB as the underlying engine. Please fetch the latest version from [Releases](https://github.com/purduedb/iPDb/releases) (i.e., `ipdb-<latest_version>.tar.gz`). To install the Python package, run:

```bash
pip install ipdb-<latest_version>.tar.gz
```

The above command will build the wheel and install iPDB as a Python package. The API is identical to the [DuckDB Python API](https://duckdb.org/docs/stable/clients/python/overview), however, this version supports all the semantic operators.  

##  Building the Platform

You can get an up-and-running instance of `iPDb` with either **Building from the source** or **Building with Docker**. 

### From the source

#### Step 1.1 - Prerequisite (Setup ONNX)

> This step is optional if you are only inferencing with LLMs (via API or local models).

First, configure the ONNX runtime library, which we'll use for pre-trained models stored in `.onnx` format. You have multiple options to get an up-and-running instance of the ONNX runtime library.

##### Build from scratch

Detailed instructions are available here [https://onnxruntime.ai/docs/build/inferencing.html](https://onnxruntime.ai/docs/build/inferencing.html).

##### Fetch a release binary

Download and extract a compatible release for your development platform from the `onnxruntime` GitHub [releases](https://github.com/microsoft/onnxruntime/releases/tag/v1.19.2).  

> The above link directs to the version `1.19.2` instead of the latest version, as this is the version tested to work (on `linux-amd64`).

For both of the above options, set the ONNX runtime installed path as an env variable.

```bash
export ONNX_INSTALL_PREFIX=<onnx_runtime_installed_path>
```

#### Step 1.2 - Prerequisite (llama.cpp)

> This step is optional if you are only inferencing with LLM APIs or pre-trained ONNX models.

First, fetch and build the `llama.cpp` library, which we'll use for inferencing local large language models. You have multiple options to get an up-and-running instance of the `llama.cpp` library.

##### Build from scratch

Clone the `llama.cpp` repository.

```bash
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
```

Build for the project using CMake (CPU build).

```bash
cmake -B build
cmake --build build --config Release -j 8
```

> `Release` builds are preferred to achieve the best inference performance.

Please refer to the detailed building guide [here](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md) in the `llama.cpp` repository for other backend (i.e. GPU), debug builds and trobleshooting.

Set the `llama.cpp` installed path as a env variable.

```bash
export LIBLLAMA_INSTALL_PREFIX="<llama_cpp_repo>/build"
```

#### Step 2 - Build our iPDb instance

`duckdb` uses a `make` script for the builds. We have extended that with additional parameters to configure `iPDb`.

```
make debug GEN=ninja -j12 CORE_EXTENSIONS='httpfs' ENABLE_PREDICT=1 PREDICTOR_IMPL=llama_cpp ENABLE_LLM_API=True DISABLE_SANITIZER=1
```

##### Build Options

- `GEN=ninja -j12` (optional, drop if you don't have ninja setup) use the ninja for build parallelization (w/ -j12 for 12 workers).
- `ENABLE_PREDICT=1` enables the ML extension.
- `PREDICTOR_IMPL=onnx` choose the internal ML platform. Available options,
  -  `onnx` - Use ONNX Runtime to infer pre-trained `.onnx` models (Step 1.1 required).
  -  `llama_cpp` - Use llama.cpp to infer LLMs in `GGUF` format.
- `ENABLE_LLM_API=1` Enable LLM calling with OpenAI API compatible APIs via the network (sets the `CORE_EXTENSIONS='httpfs'` option automatically).

Previously available `torchscript` for inferring pre-trained `pytorch` models exported with TorchScript is \[DEPRECATED\] and is not supported. 

##### API LLM Calling

The majority of publicly available remote LLMs require an API key from the respective developer to use their capabilities. `iPDb` lets users define the LLM API using the `CREATE SECRET` statement, where users can define either in-memory or persisted API keys that can be reused with different models. Please acquire the respective key and use the following SQL syntax for defining API keys.

```sql
CREATE PERSISTENT SECRET openai_key (TYPE http, bearer_token '<openai_api_key>');
CREATE PERSISTENT SECRET google_key (TYPE http, bearer_token '<google_api_key>');
```

Alternatively, the API keys can be set in the `env` variable as follows so that `iPDb` can identify them before inference. However, this limits the models to only one vendor as only one API is available.

```bash
export OPENAI_API_KEY="<api_key>"
```

> Add the above in the shell config file, i.e., `.bashrc` for permanent availability.

### With Docker

> Currently Docker script is written build only for ONNX model support. Please build from scratch to build for LLM inference.

Make sure Docker is set up correctly.

Clone the `iPDb` repository. It should include a `Dockerfile` and a `.dockerignore`.

Build the Docker image by running (we'll create an image named `iPDb`).

```bash
docker build -t iPDb
```

Run the Docker image:

```bash
docker run -it -v <absolute_path_to_data_repo>:/data --name=”iPDb_container” iPDb /bin/bash
```

For subsequent runs, you can just spin up the stopped container using the name of the container.

```bash
docker start -ai iPDb_container
```

> Both of the above commands will open an interactive shell in the Docker container. Here, we mount a data directory where we will be storing the pre-trained models. If you want just to see if the iPDb works, run the about command without `-v <absolute_path_to_mldb_repo>:/data`.

Run iPDb:
```bash
./build/debug/iPDb <your_database>
```

## Getting Started

Once you have a working iPDb instance, you can experiment with SQL queries that are capable of in-database inference.

### Simple Example

Within the iPDb shell,

- Create and populate tables with feature data (say, iris data).
-  Upload the model to the database via the `CREATE MODEL` statement.
	 ```sql
	 CREATE TABULAR MODEL iris_cls PATH `<model_path>` ON TABLE iris OUTPUT (class INTEGER); 
	 ```
-  Run a prediction query.
	```
	SELECT * FROM PREDICT(iris_cls, iris) AS p WHERE p.class = 2;
	```

Syntax tree and examples of both `CREATE MODEL` and `PREDICT` statements are available [here](https://drive.google.com/file/d/1j1qS_mJbFlXFbnh4ZLTtmB_KPRfZKG3u/view?usp=sharing) (opened via draw.io). 

### Semantive Predicate with Remote LLM

Make sure you have built `iPDb` with options that enable remote LLM calling,

- `ENABLE_PREDICT=1`
- `ENABLE_LLM_API=1`

Additionally, make sure either `SECRET`s (refer to "API LLM Calling" section) or `OPENAI_API_KEY` environment variable is set with the OpenAI API key correctly.

Within the iPDb shell,

- Create and populate tables with data (say, a `job` table with `description` column containing a job listing document).
-  Upload the model to the database via the `CREATE MODEL` statement.
	 ```sql
	 CREATE LLM MODEL o4_mini PATH 'o4-mini' ON PROMPT API 'https://api.openai.com' SECRET openai_key; 
	 ```
	 
Notice that `PATH` accepts the model name accepted by the API (e.g., `o4-mini`, `gpt-4.1`).

Furthermore, the model is uploaded as `ON PROMPT`, which results in the query execution pipeline inferring the input/output columns to the LLM from the prompt during the query execution.

Additionally, the user should set `API` to the base URL of the respective LLM. 

-  Run a prediction query.
	```sql
	SELECT * FROM LLM o4_mini (PROMPT 'extract the {s:location} and {d:salary} for job {description}', job);
	```

Here, notice that we have an additional `PROMPT` clause within the `PREDICT` statement. Inside the prompt, user can define input columns by mentioning each column with braces, i.e., `{column}` (e.g., `{descrtiption}` in the above query). Similarly, the user can define the output columns with braces in the format, `{data_type:column_name}` (e.g., `{s:location}` returns a `VARCHAR` column with location and `{d:salary}` returns a `INTEGER` column with salary).

## Additional Resources

- More details on building DuckDB from source are [here](https://duckdb.org/docs/stable/dev/building/overview).
- Follow the LibTorch tutorial on native model inference.
	- If you are new to deep-learning inference, start with beginner-friendly Python-based training and inference examples [here](https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html).
	- Introduction to native C++ model exporting and inference with TorchScript is available [here](https://pytorch.org/tutorials/advanced/cpp_export.html). The most feature-rich version of iPDb is implemented via the ONNX runtime instead of LibTorch, but this guide is far better than any ONNX examples can offer (the concepts are the same).
- The high-level duckdb execution model is explained [in the official documentation](https://duckdb.org/docs/internals/overview) and these [slides](https://15721.courses.cs.cmu.edu/spring2023/slides/22-duckdb.pdf).
- Read the source for the implementation of iPDb. **Extensive documentation of this extension is WIP.**

## Data Platforms and Runtimes

Native relational prediction operator is realized using the following systems and frameworks.

- DuckDB (1.3.0)- Relational Database [https://duckdb.org](https://duckdb.org/)
- ONNX Runtime (1.19) - Efficient generalizable deep learning runtime [https://onnx.ai](https://onnx.ai/)
- llama.cpp (latest) - Local LLM inference  [https://github.com/ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)

## DuckDB

Please refer to the Official DuckDB repository for documentation of the base source code ([https://github.com/duckdb/duckdb](https://github.com/duckdb/duckdb)).
