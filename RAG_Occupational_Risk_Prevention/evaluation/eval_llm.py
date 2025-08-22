import pathlib

PROMPT_FILE = pathlib.Path('evaluation/llm_judge_prompt.txt')

def main():
    print('🧪 LLM-as-judge — Template of prompt')
    print('Path:', PROMPT_FILE.resolve())
    print('\n=== Prompt of evaluation (overview) ===\n')
    txt = PROMPT_FILE.read_text(encoding='utf-8')
    print(txt)

if __name__ == '__main__':
    main()
