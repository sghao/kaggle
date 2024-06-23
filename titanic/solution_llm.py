import datetime
import json
import re
import fire
import sqlite3
import subprocess
import os

import titanic_utils

TRAIN_DATA_PATH = './data/train.csv'
TEST_DATA_PATH = './data/test.csv'
SUBMISSION_PATH = './data/submission.csv'

PROMPT_END_NOTE = f"""
训练数据路径为'{TRAIN_DATA_PATH}'，测试数据路径为'{TEST_DATA_PATH}'，请将预测结果保存为'{SUBMISSION_PATH}'。
"""


def execute_python_code(code):
    # Run the code and capture the output
    process = subprocess.Popen(['python', '-c', code], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()

    # Decode the output from bytes to string
    stdout = stdout.decode('utf-8')
    stderr = stderr.decode('utf-8')

    return stdout, stderr, process.returncode


def insert_result_to_database(result):
    conn = sqlite3.connect('./data/titanic_result.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS titanic_result
                    (id INTEGER PRIMARY KEY AUTOINCREMENT,
                    datetime TEXT,
                    prompt TEXT,
                    engine TEXT,
                    model TEXT,
                    gpt_options TEXT,
                    chat_completion TEXT,
                    llm_response TEXT,
                    code TEXT,
                    lines_of_code INTEGER,
                    duration_in_seconds REAL,
                    stdout TEXT,
                    stderr TEXT,
                    return_code INTEGER,
                    submission_content TEXT,
                    test_accuracy REAL)''')

    keys = list(result.keys())
    values = [result[key] for key in keys]
    keys_sql = ", ".join(keys)
    values_sql = ", ".join(["?"] * len(keys))
    c.execute(
        f"INSERT INTO titanic_result ({keys_sql}) VALUES ({values_sql})",
        values)
    conn.commit()
    conn.close()


def main_openai(prompt, model="gpt-3.5-turbo"):
    gpt_options = {
        "model": model,
        "temperature": 0.1,
        "max_tokens": 2000,
        "top_p": 1,
    }

    begin = datetime.datetime.now()
    result = {
        "datetime": begin,
        "prompt": prompt,
        "engine": "openai",
        "model": gpt_options["model"],
        "gpt_options": str(gpt_options),
    }

    from openai import OpenAI
    client = OpenAI()
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        **gpt_options,
    )
    result["chat_completion"] = chat_completion.json()

    llm_response = chat_completion.choices[0].message.content
    result["llm_response"] = llm_response

    code = llm_response.split('```python')[1].split('```')[0].strip()
    result["code"] = code

    stdout, stderr, returncode = execute_python_code(code)
    result["stdout"] = stdout
    result["stderr"] = stderr
    result["return_code"] = returncode

    if returncode == 0 and os.path.exists(SUBMISSION_PATH):
        with open(SUBMISSION_PATH) as f:
            submission_content = f.read()
        test_accuracy = titanic_utils.evaluate_submission_accuracy(SUBMISSION_PATH)
    else:
        submission_content = None
        test_accuracy = None
    result["submission_content"] = submission_content
    result["test_accuracy"] = test_accuracy
    result["lines_of_code"] = code.count('\n') + 1
    result["duration_in_seconds"] = (datetime.datetime.now() - begin).total_seconds()

    insert_result_to_database(result)


async def main_di(prompt, model="gpt-3.5-turbo", use_reflection=True):
    import yaml
    CONFIG_YAML_PATH = os.path.expanduser("~/.metagpt/config2.yaml")
    with open(CONFIG_YAML_PATH, "r") as f:
        config = yaml.safe_load(f)
    config["llm"]["model"] = model
    with open(CONFIG_YAML_PATH, "w") as f:
        yaml.safe_dump(config, f)

    begin = datetime.datetime.now()
    result = {
        "datetime": begin,
        "prompt": prompt,
        "engine": "di",
        "model": model,
    }

    from metagpt.roles.di.data_interpreter import DataInterpreter
    from metagpt.tools.tool_recommend import TypeMatchToolRecommender

    di = DataInterpreter(use_reflection=True, tool_recommender=TypeMatchToolRecommender(tools=["<all>"]))
    message = await di.run(prompt)
    result["chat_completion"] = message.dump()
    content = message.content
    result["llm_response"] = content

    code_blocks = []
    plan = re.search(r'## Current Plan\n(.*?)\n## ', content, re.DOTALL).group(1).strip()
    plan = json.loads(plan)
    for task in plan:
        code_blocks.append(task["code"])
    code = "\n\n".join(code_blocks)

    result["code"] = code
    result["lines_of_code"] = code.count('\n') + 1
    result["duration_in_seconds"] = (datetime.datetime.now() - begin).total_seconds()

    if os.path.exists(SUBMISSION_PATH):
        with open(SUBMISSION_PATH) as f:
            submission_content = f.read()
        test_accuracy = titanic_utils.evaluate_submission_accuracy(SUBMISSION_PATH)
    else:
        submission_content = None
        test_accuracy = None
    result["submission_content"] = submission_content
    result["test_accuracy"] = test_accuracy

    insert_result_to_database(result)


async def main(prompt_file, engine="openai", model="gpt-3.5-turbo"):
    """
    Run the Titanic challenge with the given prompt file and engine.

    Args:
        prompt_file: The path to the prompt file.
        engine: The engine to use, either "openai" or "di".
        model: The model to use, choices=["gpt-3.5-turbo", "gpt-4o", "gpt-4-turbo"].
    """
    print(f"Using {engine} engine with model {model}, reading from {prompt_file}...")
    with open(prompt_file) as f:
        prompt = f.read()
    prompt = f"{prompt}{PROMPT_END_NOTE}"

    if os.path.exists(SUBMISSION_PATH):
        os.remove(SUBMISSION_PATH)

    if engine == "openai":
        main_openai(prompt, model)
    elif engine == "di":
        await main_di(prompt, model)


if __name__ == '__main__':
    fire.Fire(main)
