import time
import numpy as np
import os
import dotenv
import openai
import backoff
import string
import random
from collections import defaultdict

dotenv.load_dotenv(dotenv.find_dotenv())
openai.api_key = os.getenv("OPENAI_API_KEY")

from textflint.input.model import FlintModel
from textflint.input.component import Sample
from textflint.input.model.metrics.metrics import accuracy_score as Accuracy

RATE_LIMIT_PER_MIN = 3500
DELAY = 60 / RATE_LIMIT_PER_MIN

@backoff.on_exception(backoff.expo, (openai.error.RateLimitError, openai.error.APIError), max_time=60)
def _chatcompletions_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)

class OpenAIFlint(FlintModel):
    """Model wrapper for ChatGPT (gpt-3.5-turbo) using OpenAI. Default task is Sentiment Analysis."""

    def __init__(self, model_name, batch_size=2000):
        self.model = model_name
        self.label2id = {'pos': 1, 'neg': 0, 'positive': 1, 'negative': 0}
        self.batch_size = batch_size
        self.responses = defaultdict(list)
        self.current_prefix = ''
    
    def __call__(self, batch_inputs, max_tokens=10):
        """Return array of batch outputs (in ID form) from the model."""
        batch_outputs = []

        example = {
            'x': "Robert DeNiro plays the most unbelievably intelligent illiterate of all time. \
                This movie is so wasteful of talent, it is truly disgusting. The script is unbelievable. \
                    The dialog is unbelievable. Jane Fonda's character is a caricature of herself, and not a funny one.\
                          The movie moves at a snail's pace, is photographed in an ill-advised manner, and is insufferably preachy.\
                              It also plugs in every cliche in the book. Swoozie Kurtz is excellent in a supporting role,\
                                  but so what?<br /><br />Equally annoying is this new IMDB rule of requiring ten lines for every review. \
                                    When a movie is this worthless, it doesn't require ten lines of text to let other readers know that it is a waste of time and tape. Avoid this movie.",
            'y': 'neg'
        }

        for data_point in batch_inputs:
            try:
                response = self._delayed_completion(DELAY, data_point=data_point, example=example, max_tokens=max_tokens)
                self.responses[self.current_prefix].append(response)
                content = response.choices[0].message.content.lower()
                if content in self.label2id.keys():
                    batch_outputs.append(self.label2id[content])
                else:
                    print(f'Invalid model response: \"{content}\". The data_point was: {data_point}')
                    batch_outputs.append(-1)
            except openai.error.InvalidRequestError as e:
                print(f'Encountered InvalidRequestError for data_point: {data_point}. Skipped and counted as incorrect example.')
                batch_outputs.append(-1)
        return batch_outputs
    
    def _get_response(self, data_point, example, max_tokens):
        messages = [
            {"role": "system", "content": "You are performing sentiment analysis. Respond with only with \"pos\" or \"neg\"."},
            {"role": "user", "content": example['x']},
            {"role": "assistant", "content": example['y']},
            {"role": "user", "content": data_point['x']}
        ]
        response = _chatcompletions_with_backoff(
            model=self.model,
            messages=messages,
            temperature=0
        )

        return response
    
    def _delayed_completion(self, delay_in_seconds: float = 1, **kwargs):
        time.sleep(delay_in_seconds)
        return self._get_response(**kwargs)

    def unzip_samples(self, data_samples: list[Sample]):
        x, y = [], []

        for sample in data_samples:
            x.append(sample)
            y.append(self.label2id[sample['y']])
        
        return x, y
    
    def evaluate(self, data_samples: list[Sample], prefix: str = ''):
        self.current_prefix = prefix
        outputs, labels = [], []

        i = 0
        while i < len(data_samples):
            batch_samples = data_samples[i: i + self.batch_size]
            batch_inputs, batch_labels = self.unzip_samples(batch_samples)
            labels += batch_labels
            outputs += self.__call__(batch_inputs)
            i += self.batch_size

        metrics_rst = {
            prefix + '_Accuracy': Accuracy(np.array(outputs), np.array(labels))
        }

        return metrics_rst

class OpenAI_NLI_Flint(OpenAIFlint):
    def __init__(self, model_name, batch_size=2000):
        super().__init__(model_name=model_name, batch_size=batch_size)
        self.label2id = {
            'entailment': 1,
            'neutral': 2,
            'contradiction': 3
        }
    
    def _get_response(self, data_point, example):
        assert isinstance(data_point, dict)
        # Note: the model responds with 1-based index.

        messages=[
            {"role": "system", "content": f"You are performing Natural Language Inference. When given a premise and \
             hypothesis, respond with \"entailment\", \"neutral\", or \"contradiction\" and nothing else."},
            {"role": "user", "content": f"Premise: {example['premise']}\nHypothesis: {example['hypothesis']}"},
            {"role": "assistant", "content": f"{example['y']}"},
            {"role": "user", "content": f"Premise: {data_point['premise']}\nHypothesis: {data_point['hypothesis']}"},
        ]
        return _chatcompletions_with_backoff(
            model=self.model,
            messages=messages,
            temperature=0
        )

    def __call__(self, batch_inputs):
        batch_outputs = []

        example = {
            'premise': 'Two women are embracing while holding to go packages.',
            'hypothesis': 'The sisters are hugging goodbye while holding to go packages after just eating lunch.',
            'y': 'neutral'
        }

        for data_point in batch_inputs:
            try:
                response = super()._delayed_completion(DELAY, data_point=data_point, example=example)
                self.responses[self.current_prefix].append(response)
                content = response.choices[0].message.content
                content = content.lower().translate(str.maketrans('', '', string.punctuation)) 
                if content in self.label2id.keys():
                    batch_outputs.append(self.label2id[content])
                else:
                    print(f'Invalid model response: \"{content}\". The data_point was: {data_point}')
                    batch_outputs.append(-1)
            except openai.error.InvalidRequestError as e:
                print(f'Encountered InvalidRequestError for data_point: {data_point}. Skipped and counted as incorrect example.')
                batch_outputs.append(-1)
        
        return batch_outputs

    def unzip_samples(self, data_samples: list[Sample]):
        x, y = [], []
        for sample in data_samples:
            x.append({'premise': sample['premise'], 'hypothesis': sample['hypothesis'], 'y': sample['y']})
            y.append(self.label2id[sample['y']])
        return x, y

    def evaluate(self, data_samples: list[Sample], prefix: str = ''):
        return super().evaluate(data_samples, prefix)

class OpenAI_MRC_Plus_Flint(OpenAIFlint):
    def __init__(self, model_name, batch_size=2000):
        super().__init__(model_name=model_name, batch_size=batch_size)
        self.responses = defaultdict(list)

    def __call__(self, batch_inputs):
        batch_outputs = []

        for prompt in batch_inputs:
            try:
                response = super()._delayed_completion(DELAY, prompt=prompt)
                self.responses[self.current_prefix].append(response)

                # Note: the model responds with 1-based index.
                content = response.choices[0].message.content
                
                if content[0].isdigit():
                    batch_outputs.append(int(content[0]))
                elif content[1].isdigit():
                    batch_outputs.append(int(content[1]))
                else:
                    print(f'Invalid model response: \"{content}\".\n\tThe prompt was: {prompt}')
                    batch_outputs.append(-1)
            except openai.error.InvalidRequestError as e:
                print(f'Encountered InvalidRequestError for prompt: {prompt}. Skipped and counted as incorrect example.')
                batch_outputs.append(-1)
            
        return batch_outputs

    def _get_response(self, prompt):
        assert isinstance(prompt, dict)

        # Note: the model responds with 1-based index.
        answers = '\n'.join(f'[{i}] {a}' for i, a in enumerate(prompt['answers']))
        messages = [
            {"role": "system", "content": "You are performing Machine Reading Comprehension. \
             Respond only with the index corresponding to the provided answers and nothing else."},
            {"role": "user", "content": f"Context: {prompt['context']}\n \
                                        Question: {prompt['question']}\n \
                                        Answers:\n" + answers
            }
        ]

        return _chatcompletions_with_backoff(
            model=self.model,
            messages=messages,
            temperature=0
        )

    def unzip_samples(self, data_samples):
        x, y = [], []

        for sample in data_samples:
            input = {
                'context': sample['context'],
                'question': sample['question'],
                'answers': sample['answers'],
                'answer_choices': sample['answer_choices']
            }
            x.append(input)
            y.append(sample['label'])
        
        return x, y
    
    def evaluate(self, data_samples: list[Sample], prefix: str = ''):
        return super().evaluate(data_samples, prefix)

