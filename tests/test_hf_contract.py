import torch
from mfgds.data import Example
from mfgds.models import HuggingFaceLMM


class Inputs(dict):
    def to(self, device):
        return self


class FakeProcessor:
    def apply_chat_template(self, messages, **kwargs):
        assert kwargs['tokenize'] and kwargs['add_generation_prompt']
        assert len(messages) == 3
        assert messages[0]['content'][0]['type'] == 'image'
        assert messages[1]['content'][0]['text'] == 'demo answer'
        assert messages[-1]['content'][-1]['text'].startswith('test question')
        assert 'gold secret' not in str(messages)
        return Inputs(input_ids=torch.tensor([[10, 11, 12]]))

    def batch_decode(self, ids, **kwargs):
        assert ids.tolist() == [[20, 21]]
        return ['generated answer']


class FakeModel:
    device = 'cpu'

    def generate(self, **kwargs):
        assert kwargs['do_sample'] is False
        assert kwargs['max_new_tokens'] == 4
        return torch.tensor([[10, 11, 12, 20, 21]])


def test_hf_contract_without_weights():
    model = HuggingFaceLMM.__new__(HuggingFaceLMM)
    model.processor, model.model, model.max_new_tokens = FakeProcessor(), FakeModel(), 4
    query = Example('q', 'test question', 'gold secret').query()
    demos = [Example('d', 'demo question', 'demo answer', 'demo.png')]
    assert model.count_tokens(query, demos) == 3
    prediction = model.predict(query, demos)
    assert prediction.text == 'generated answer'
    assert prediction.input_tokens == 3 and prediction.output_tokens == 2
