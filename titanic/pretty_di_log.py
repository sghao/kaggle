import ast
import fire

def main(filename):
    from_filepath = f"logs/{filename}.txt"
    to_filepath = f"logs/{filename}-pretty.txt"

    with open(from_filepath, 'r') as from_file, open(to_filepath, 'w') as to_file:
        for line in from_file:
            if "metagpt.provider.base_llm:aask:149" in line:
                index = line.index("[")
                to_file.write(line[:index] + "\n")
                messages = line[index:]
                messages = ast.literal_eval(messages)
                for message in messages:
                    content = message["content"]
                    del message["content"]
                    to_file.write("\t" + str(message) + "\n")
                    to_file.write("\t" + content + "\n")
            else:
                to_file.write(line)


if __name__ == "__main__":
    fire.Fire(main)
