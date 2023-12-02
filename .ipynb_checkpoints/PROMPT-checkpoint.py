B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"


DEFAULT_SYSTEM_PROMPT = "你是一个诚实而优秀的中国人助手。"
PROMPT_DICT = {
    "prompt_input": (
            f"{B_INST}{B_SYS}{DEFAULT_SYSTEM_PROMPT}{E_SYS}" + "{input}{instruction}" + f"{E_INST}"
    ),
    "prompt_no_input": (
            f"{B_INST}{B_SYS}{DEFAULT_SYSTEM_PROMPT}{E_SYS}" + "{instruction}" + f"{E_INST}"
    ),
}

'''
PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}
'''