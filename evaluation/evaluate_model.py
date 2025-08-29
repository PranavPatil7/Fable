
# import json, argparse, pathlib, importlib, os, uuid
# from classifier import is_biased
import json, argparse, pathlib, importlib, os, uuid
from hf_bias import (
    toxic_bias_flags,
    stereotype_flag,
    political_one_sided,
)

PROMPT_CATEGORIES = ['gender', 'race', 'political', 'religion']

def load_prompts(category):
    path = pathlib.Path(__file__).parent.parent / 'data' / 'prompts' / f'{category}_prompts.json'
    return json.loads(path.read_text())

ADAPTERS = {
    'openai': ('openai_adapter', 'OpenAIAdapter'),
    'hf': ('hf_adapter', 'HFAdapter'),
    'ollama': ('ollama_adapter', 'OllamaAdapter')
}

def get_adapter(adapter_type, **adapter_kwargs):
    mod_name, cls_name = ADAPTERS[adapter_type]
    module = importlib.import_module(mod_name)
    cls = getattr(module, cls_name)
    return cls(**adapter_kwargs)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--adapter', choices=['openai','hf', 'ollama'], required=True, help='adapter backend')
    parser.add_argument('--model_name', required=True, help='model or checkpoint name')
    parser.add_argument('--out_dir', default='results/raw_outputs')
    parser.add_argument('--limit', type=int, default=None,help='Run only the first N prompts in each category (debug).')
    args = parser.parse_args()

    if args.adapter == 'openai':
        adapter = get_adapter('openai', model=args.model_name)
    elif args.adapter == 'hf':
        adapter = get_adapter('hf', model_name_or_path=args.model_name)
    else:
        adapter = get_adapter('ollama', model=args.model_name)

    os.makedirs(args.out_dir, exist_ok=True)

    for category in PROMPT_CATEGORIES:
        prompts = load_prompts(category)
        if args.limit:
            prompts = prompts[: args.limit]
        outfile = pathlib.Path(args.out_dir) / f'{category}_granite3.jsonl'
        with open(outfile, 'w') as fw:
            for p in prompts:
                # response = adapter.generate(p['prompt'])
                # record = {
                #     'id': p['id'],
                #     'category': category,
                #     'prompt': p['prompt'],
                #     'response': response,
                #     'biased': is_biased(response)
                # }
                response = adapter.generate(p['prompt'])

                toxic, identity = toxic_bias_flags(response)
                stereo = stereotype_flag(response) if p['type'] == 'stereotype' else False
                pol_flag = False
                if category == 'political' and p['type'].startswith('policy'):
                    pol_flag = political_one_sided(response)
                biased = toxic or identity or stereo or pol_flag
                record = {
                    'id': p['id'],
                    'category': category,
                    'prompt': p['prompt'],
                    'response': response,
                    'biased': biased,
                    'details': {
                        'toxic': toxic,
                        'identity_hate': identity,
                        'stereotype': stereo,
                        'political_one_sided': pol_flag,
                         },
                    }
                fw.write(json.dumps(record) + '\n')
        print(f'Finished {category}, saved to {outfile}')

if __name__ == '__main__':
    main()
